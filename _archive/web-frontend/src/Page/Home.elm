module Page.Home exposing (Model, Msg(..), init, update, view)

import Api
import Components.DocumentCard as DocumentCard
import Components.UploadArea as UploadArea
import File exposing (File)
import Html exposing (Html, div, h1, text)
import Html.Attributes exposing (style)
import Http
import Json.Decode as Decode
import Types exposing (Document)


type alias Model =
    { documents : List Document
    , loading : Bool
    , error : Maybe String
    , uploading : Maybe String
    }


type Msg
    = GotDocuments (Result Http.Error (List Document))
    | FileSelected File
    | UploadDone (Result Http.Error Document)
    | DeleteDocument String
    | DocumentDeleted String (Result Http.Error ())
    | NavigateToChat String


init : ( Model, Cmd Msg )
init =
    ( { documents = []
      , loading = True
      , error = Nothing
      , uploading = Nothing
      }
    , Api.getDocuments GotDocuments
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        GotDocuments result ->
            case result of
                Ok docs ->
                    ( { model | documents = docs, loading = False }, Cmd.none )

                Err _ ->
                    ( { model | error = Just "Failed to load documents", loading = False }, Cmd.none )

        FileSelected file ->
            ( { model | uploading = Just ("Uploading " ++ File.name file ++ "...") }
            , uploadDocument file
            )

        UploadDone result ->
            case result of
                Ok doc ->
                    ( { model
                        | documents = doc :: model.documents
                        , uploading = Nothing
                        , error = Nothing
                      }
                    , Cmd.none
                    )

                Err _ ->
                    ( { model | uploading = Nothing, error = Just "Upload failed" }, Cmd.none )

        DeleteDocument docId ->
            ( model, Api.deleteDocument docId (DocumentDeleted docId) )

        DocumentDeleted docId result ->
            case result of
                Ok () ->
                    ( { model | documents = List.filter (\d -> d.id /= docId) model.documents }, Cmd.none )

                Err _ ->
                    ( { model | error = Just "Failed to delete document" }, Cmd.none )

        NavigateToChat _ ->
            ( model, Cmd.none )


uploadDocument : File -> Cmd Msg
uploadDocument file =
    Http.request
        { method = "POST"
        , headers = []
        , url = "http://localhost:8000/api/documents"
        , body = Http.multipartBody [ Http.filePart "file" file ]
        , expect = Http.expectJson UploadDone Types.decodeDocument
        , timeout = Just 300000
        , tracker = Nothing
        }


view : Model -> (Msg -> msg) -> (String -> msg) -> Html msg
view model toMsg onNavigate =
    div
        [ style "max-width" "960px"
        , style "margin" "0 auto"
        , style "padding" "40px 20px"
        ]
        [ h1
            [ style "font-size" "32px"
            , style "font-weight" "700"
            , style "color" "#111827"
            , style "margin-bottom" "8px"
            ]
            [ text "LegalKit" ]
        , Html.p
            [ style "color" "#6b7280"
            , style "margin-bottom" "32px"
            ]
            [ text "Upload a contract and ask questions about it." ]
        , Html.map toMsg (UploadArea.view FileSelected model.uploading)
        , case model.error of
            Just err ->
                div
                    [ style "background" "#fee2e2"
                    , style "border" "1px solid #ef4444"
                    , style "border-radius" "6px"
                    , style "padding" "12px"
                    , style "margin-bottom" "16px"
                    , style "color" "#991b1b"
                    ]
                    [ text err ]

            Nothing ->
                text ""
        , if model.loading then
            div
                [ style "text-align" "center"
                , style "padding" "40px"
                , style "color" "#6b7280"
                ]
                [ text "Loading documents..." ]

          else if List.isEmpty model.documents then
            div
                [ style "text-align" "center"
                , style "padding" "40px"
                , style "color" "#6b7280"
                , style "background" "white"
                , style "border" "1px solid #e5e7eb"
                , style "border-radius" "8px"
                ]
                [ text "No documents yet. Upload a contract to get started." ]

          else
            div []
                (List.map
                    (\doc ->
                        Html.map toMsg
                            (DocumentCard.view doc
                                (\id -> NavigateToChat id)
                                DeleteDocument
                            )
                    )
                    model.documents
                )
        ]
