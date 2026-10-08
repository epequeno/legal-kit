module Types exposing
    ( Document
    , AnswerReport
    , Citation
    , ApiKeyStatus
    , DocumentOutline
    , OutlineSection
    , decodeDocument
    , decodeDocumentList
    , decodeAnswerReport
    , decodeAnswerReportList
    , decodeApiKeyStatus
    , decodeDocumentOutline
    )

import Json.Decode as Decode exposing (Decoder)


type alias Document =
    { id : String
    , filename : String
    , originalPath : String
    , sizeBytes : Int
    , nChunks : Int
    , ingestedAt : String
    }


type alias Citation =
    { nodeId : String
    , nodePath : List String
    , quote : String
    , charStart : Int
    , charEnd : Int
    , confidence : Float
    }


type alias AnswerReport =
    { question : String
    , answer : String
    , citations : List Citation
    , docId : String
    , elapsedSeconds : Float
    }


type alias ApiKeyStatus =
    { openrouter : Bool
    , openai : Bool
    , anthropic : Bool
    , ready : Bool
    , activeModel : String
    , localEmbeddings : Bool
    , localEntailment : Bool
    , localReranker : Bool
    }


decodeDocument : Decoder Document
decodeDocument =
    Decode.map6 Document
        (Decode.field "id" Decode.string)
        (Decode.field "filename" Decode.string)
        (Decode.field "original_path" Decode.string)
        (Decode.field "size_bytes" Decode.int)
        (Decode.field "n_chunks" Decode.int)
        (Decode.field "ingested_at" Decode.string)


decodeDocumentList : Decoder (List Document)
decodeDocumentList =
    Decode.list decodeDocument


decodeCitation : Decoder Citation
decodeCitation =
    Decode.map6 Citation
        (Decode.field "node_id" Decode.string)
        (Decode.field "node_path" (Decode.list Decode.string))
        (Decode.field "quote" Decode.string)
        (Decode.field "char_start" Decode.int)
        (Decode.field "char_end" Decode.int)
        (Decode.field "confidence" Decode.float)


decodeAnswerReport : Decoder AnswerReport
decodeAnswerReport =
    Decode.map5 AnswerReport
        (Decode.field "question" Decode.string)
        (Decode.field "answer" Decode.string)
        (Decode.field "citations" (Decode.list decodeCitation))
        (Decode.field "doc_id" Decode.string)
        (Decode.field "elapsed_seconds" Decode.float)


decodeAnswerReportList : Decoder (List AnswerReport)
decodeAnswerReportList =
    Decode.list decodeAnswerReport


type alias OutlineSection =
    { id : String
    , title : String
    , nodeType : String
    }


type alias DocumentOutline =
    { docId : String
    , totalSections : Int
    , topSections : List OutlineSection
    , crossRefsCount : Int
    , subsectionCount : Int
    }


decodeOutlineSection : Decoder OutlineSection
decodeOutlineSection =
    Decode.map3 OutlineSection
        (Decode.field "id" Decode.string)
        (Decode.field "title" Decode.string)
        (Decode.field "node_type" Decode.string)


decodeDocumentOutline : Decoder DocumentOutline
decodeDocumentOutline =
    Decode.map5 DocumentOutline
        (Decode.field "doc_id" Decode.string)
        (Decode.field "total_sections" Decode.int)
        (Decode.field "top_sections" (Decode.list decodeOutlineSection))
        (Decode.field "cross_refs_count" Decode.int)
        (Decode.field "subsection_count" Decode.int)


decodeApiKeyStatus : Decoder ApiKeyStatus
decodeApiKeyStatus =
    Decode.map8
        (\openrouter openai anthropic ready activeModel localEmbed localEntail localReranker ->
            { openrouter = openrouter
            , openai = openai
            , anthropic = anthropic
            , ready = ready
            , activeModel = activeModel
            , localEmbeddings = localEmbed
            , localEntailment = localEntail
            , localReranker = localReranker
            }
        )
        (Decode.field "openrouter" Decode.bool)
        (Decode.field "openai" Decode.bool)
        (Decode.field "anthropic" Decode.bool)
        (Decode.field "ready" Decode.bool)
        (Decode.field "active_model" Decode.string)
        (Decode.field "local_embeddings" Decode.bool)
        (Decode.field "local_entailment" Decode.bool)
        (Decode.field "local_reranker" Decode.bool)
