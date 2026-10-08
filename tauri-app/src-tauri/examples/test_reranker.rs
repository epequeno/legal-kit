fn main() {
    use std::path::Path;
    use std::time::Instant;

    let model_path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/legal-reranker/final"));
    legalkit_lib::pipeline::rerank_local::ensure_loaded(model_path).unwrap();

    let q = "Does this contract include an exclusivity agreement?";
    let passages = vec![
        "Throughout the Term, neither party shall sell competing products without written consent.",
        "The governing law shall be the laws of the State of Delaware.",
        "This Agreement may be terminated by either party upon 30 days written notice.",
        "Seller grants Buyer an exclusive right to distribute Products in the Territory.",
        "Payment shall be due within 30 days of invoice date.",
        "Neither party shall disclose Confidential Information to third parties.",
        "The term of this Agreement shall commence on the Effective Date and continue for two years.",
        "Buyer shall not purchase competing products from any third party during the Term.",
        "Intellectual property developed under this Agreement shall be jointly owned.",
        "Force majeure events shall excuse performance obligations for the duration of such event.",
        "All notices shall be in writing and delivered by certified mail.",
        "This Agreement constitutes the entire agreement between the parties.",
        "No amendment to this Agreement shall be valid unless in writing and signed by both parties.",
        "Indemnification shall cover all third-party claims arising from breach of this Agreement.",
        "The parties agree to submit disputes to binding arbitration in New York.",
        "Royalties shall be calculated at 5% of net sales and paid quarterly.",
        "The exclusive territory is defined as the continental United States and Canada.",
        "Supplier shall maintain minimum inventory levels as specified in Exhibit A.",
        "Either party may audit the other's books with 10 days written notice.",
        "This Agreement shall be binding upon successors and assigns of both parties.",
    ];

    let t0 = Instant::now();
    let pairs: Vec<(String, usize)> = passages.iter().enumerate()
        .map(|(i, p)| (p.to_string(), i)).collect();
    let ranked = legalkit_lib::pipeline::rerank_local::rerank(q, &pairs).unwrap();
    let elapsed = t0.elapsed();

    println!("Reranked {} passages in {:.0}ms", passages.len(), elapsed.as_millis());
    println!();
    println!("Top 5:");
    for (orig_idx, score) in ranked.iter().take(5) {
        println!("  [{:.4}] {}", score, &passages[*orig_idx][..60]);
    }
}
