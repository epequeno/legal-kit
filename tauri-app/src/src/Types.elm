module Types exposing
    ( AskResult
    , Citation
    , Claim
    , Confidence
    , Document
    , decodeAskResult
    , decodeCitation
    , decodeClaim
    , decodeConfidence
    , decodeDocument
    , decodeDocumentList
    )

import Json.Decode as Decode exposing (Decoder)


type alias Document =
    { id : String
    , filename : String
    , sizeBytes : Int
    , nChunks : Int
    , ingestedAt : String
    }


type alias Citation =
    { sectionId : String
    , sourceText : String
    , status : String
    , entailment : String
    , entailmentScore : Float
    }


type alias Claim =
    { text : String
    , isGap : Bool
    , hasHedging : Bool
    , confidence : Confidence
    , citations : List Citation
    }


type alias Confidence =
    { level : String
    , score : Float
    }


type alias AskResult =
    { answer : String
    , claims : List Claim
    , overallConfidence : Confidence
    , gaps : List String
    , elapsedSeconds : Float
    , docId : String
    }



-- DECODERS


decodeDocument : Decoder Document
decodeDocument =
    Decode.map5 Document
        (Decode.field "id" Decode.string)
        (Decode.field "filename" Decode.string)
        (Decode.field "size_bytes" Decode.int)
        (Decode.field "n_chunks" Decode.int)
        (Decode.field "ingested_at" Decode.string)


decodeDocumentList : Decoder (List Document)
decodeDocumentList =
    Decode.list decodeDocument


decodeCitation : Decoder Citation
decodeCitation =
    Decode.map5 Citation
        (Decode.field "section_id" Decode.string)
        (Decode.field "source_text" Decode.string)
        (Decode.field "status" Decode.string)
        (Decode.field "entailment" Decode.string)
        (Decode.field "entailment_score" Decode.float)


decodeConfidence : Decoder Confidence
decodeConfidence =
    Decode.map2 Confidence
        (Decode.field "level" Decode.string)
        (Decode.field "score" Decode.float)


decodeClaim : Decoder Claim
decodeClaim =
    Decode.map5 Claim
        (Decode.field "text" Decode.string)
        (Decode.field "is_gap" Decode.bool)
        (Decode.field "has_hedging" Decode.bool)
        (Decode.field "confidence" decodeConfidence)
        (Decode.field "citations" (Decode.list decodeCitation))


decodeAskResult : Decoder AskResult
decodeAskResult =
    Decode.map6 AskResult
        (Decode.field "answer" Decode.string)
        (Decode.field "claims" (Decode.list decodeClaim))
        (Decode.field "overall_confidence" decodeConfidence)
        (Decode.field "gaps" (Decode.list Decode.string))
        (Decode.field "elapsed_seconds" Decode.float)
        (Decode.field "doc_id" Decode.string)
