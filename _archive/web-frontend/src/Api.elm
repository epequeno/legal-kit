module Api exposing
    ( deleteDocument
    , getDocument
    , getDocuments
    , getHealth
    )

import Http
import Json.Decode as Decode
import Types exposing (Document, decodeDocument, decodeDocumentList)


baseUrl : String
baseUrl =
    "http://localhost:8000/api"


getHealth : (Result Http.Error String -> msg) -> Cmd msg
getHealth toMsg =
    Http.get
        { url = baseUrl ++ "/health"
        , expect = Http.expectString toMsg
        }


getDocuments : (Result Http.Error (List Document) -> msg) -> Cmd msg
getDocuments toMsg =
    Http.get
        { url = baseUrl ++ "/documents"
        , expect = Http.expectJson toMsg decodeDocumentList
        }


getDocument : String -> (Result Http.Error Document -> msg) -> Cmd msg
getDocument docId toMsg =
    Http.get
        { url = baseUrl ++ "/documents/" ++ docId
        , expect = Http.expectJson toMsg decodeDocument
        }


deleteDocument : String -> (Result Http.Error () -> msg) -> Cmd msg
deleteDocument docId toMsg =
    Http.request
        { method = "DELETE"
        , headers = []
        , url = baseUrl ++ "/documents/" ++ docId
        , body = Http.emptyBody
        , expect = Http.expectWhatever toMsg
        , timeout = Nothing
        , tracker = Nothing
        }
