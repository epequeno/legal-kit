//! Stage 0: Extract plain text from TXT / PDF / DOCX.

use anyhow::{bail, Result};
use std::path::Path;

pub fn extract_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "txt" => extract_txt(path),
        "pdf" => extract_pdf(path),
        "docx" => extract_docx(path),
        other => bail!("Unsupported file type: .{other}"),
    }
}

fn extract_txt(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    // Try UTF-8 first, then Latin-1 fallback (common in older legal docs)
    match String::from_utf8(bytes.clone()) {
        Ok(s) => Ok(s),
        Err(_) => Ok(bytes.iter().map(|&b| b as char).collect()),
    }
}

fn extract_pdf(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let text = pdf_extract::extract_text_from_mem(&bytes)?;
    Ok(text)
}

fn extract_docx(path: &Path) -> Result<String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;
    use std::io::Read;

    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    // word/document.xml holds the body text
    let xml = {
        let mut entry = archive
            .by_name("word/document.xml")
            .map_err(|_| anyhow::anyhow!("Not a valid .docx file (missing word/document.xml)"))?;
        let mut buf = String::new();
        entry.read_to_string(&mut buf)?;
        buf
    };

    // Pull text out of <w:t> elements, insert newlines at <w:p> boundaries
    let mut reader = Reader::from_str(&xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();
    let mut in_paragraph = false;
    let mut para_parts: Vec<String> = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                match e.name().as_ref() {
                    b"w:p" => {
                        in_paragraph = true;
                        para_parts.clear();
                    }
                    b"w:br" => {
                        // Line break inside paragraph
                        if in_paragraph {
                            para_parts.push("\n".to_string());
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                if e.name().as_ref() == b"w:p" {
                    let para = para_parts.join("");
                    if !para.trim().is_empty() {
                        text_parts.push(para);
                    }
                    in_paragraph = false;
                }
            }
            Ok(Event::Text(e)) => {
                if in_paragraph {
                    if let Ok(s) = e.unescape() {
                        para_parts.push(s.into_owned());
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => bail!("XML parse error in docx: {e}"),
            _ => {}
        }
        buf.clear();
    }

    Ok(text_parts.join("\n"))
}
