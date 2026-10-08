module Page.Home exposing (Model, Msg(..), init, update, view)

import Components.DocumentCard as DocumentCard
import Html exposing (Html, button, div, span, text)
import Html.Attributes exposing (class, style)
import Html.Events exposing (onClick)
import Json.Decode as Decode
import Json.Encode as Encode
import Ports
import Types exposing (ApiKeyStatus, Document, decodeApiKeyStatus, decodeDocument, decodeDocumentList)


type alias Model =
    { documents : List Document
    , loading : Bool
    , ingesting : Bool
    , error : Maybe String
    , apiKeys : Maybe ApiKeyStatus
    }


type Msg
    = OpenFilePicker
    | FileSelected String
    | GotIngestResult Decode.Value
    | GotListResult Decode.Value
    | GotApiKeyStatus Decode.Value
    | GotError String
    | DeleteDocument String
    | GotDeleteResult Decode.Value
    | NavigateToChat String


init : ( Model, Cmd Msg )
init =
    ( { documents = []
      , loading = True
      , ingesting = False
      , error = Nothing
      , apiKeys = Nothing
      }
    , Cmd.batch
        [ Ports.invokeListDocuments ()
        , Ports.invokeGetApiKeyStatus ()
        ]
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        OpenFilePicker ->
            ( model, Ports.openFilePicker () )

        FileSelected path ->
            ( { model | ingesting = True, error = Nothing }
            , Ports.invokeIngestDocument path
            )

        GotIngestResult val ->
            case Decode.decodeValue decodeDocument val of
                Ok doc ->
                    ( { model
                        | documents = doc :: model.documents
                        , ingesting = False
                      }
                    , Cmd.none
                    )

                Err e ->
                    ( { model
                        | ingesting = False
                        , error = Just ("Ingest failed: " ++ Decode.errorToString e)
                      }
                    , Cmd.none
                    )

        GotListResult val ->
            case Decode.decodeValue decodeDocumentList val of
                Ok docs ->
                    ( { model | documents = docs, loading = False }, Cmd.none )

                Err _ ->
                    ( { model | loading = False }, Cmd.none )

        GotApiKeyStatus val ->
            case Decode.decodeValue decodeApiKeyStatus val of
                Ok status ->
                    ( { model | apiKeys = Just status }, Cmd.none )

                Err _ ->
                    ( model, Cmd.none )

        GotError err ->
            ( { model | ingesting = False, error = Just err }, Cmd.none )

        DeleteDocument docId ->
            ( model, Ports.invokeDeleteDocument docId )

        GotDeleteResult _ ->
            ( model, Ports.invokeListDocuments () )

        NavigateToChat _ ->
            ( model, Cmd.none )


view : Model -> (Msg -> msg) -> (String -> msg) -> msg -> Bool -> Html msg
view model toMsg _ onSettings settingsActive =
    div
        [ style "width" "220px"
        , style "background" "#141414"
        , style "border-right" "1px solid #1e1e1e"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "flex-shrink" "0"
        , style "height" "100%"
        ]
        [ -- App header
          div
            [ style "padding" "16px 14px 14px"
            , style "border-bottom" "1px solid #1e1e1e"
            , style "display" "flex"
            , style "align-items" "center"
            , style "gap" "9px"
            ]
            [ -- Small icon mark
              div
                [ style "width" "22px"
                , style "height" "22px"
                , style "background" "#1e3a5f"
                , style "border" "1px solid #2563eb"
                , style "border-radius" "5px"
                , style "display" "flex"
                , style "align-items" "center"
                , style "justify-content" "center"
                , style "flex-shrink" "0"
                ]
                [ span [ style "font-size" "11px", style "font-weight" "900", style "color" "#60a5fa" ] [ text "L" ] ]
            , span
                [ style "font-size" "13px"
                , style "font-weight" "800"
                , style "color" "#888"
                , style "letter-spacing" "-0.2px"
                ]
                [ text "LegalKit" ]
            ]

        -- Error banner
        , case model.error of
            Just err ->
                div
                    [ style "margin" "8px"
                    , style "background" "#2d0a0a"
                    , style "border" "1px solid #7f1d1d"
                    , style "border-radius" "6px"
                    , style "padding" "8px 10px"
                    , style "font-size" "11px"
                    , style "color" "#fca5a5"
                    , style "line-height" "1.4"
                    ]
                    [ text err ]

            Nothing ->
                text ""

        -- Section label
        , div
            [ style "padding" "12px 14px 6px"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "space-between"
            ]
            [ span
                [ style "font-size" "10px"
                , style "font-weight" "700"
                , style "color" "#2e2e2e"
                , style "letter-spacing" "0.08em"
                , style "text-transform" "uppercase"
                ]
                [ text "Contracts" ]
            , if not (List.isEmpty model.documents) then
                span
                    [ style "font-size" "10px"
                    , style "color" "#2a2a2a"
                    , style "font-weight" "600"
                    ]
                    [ text (String.fromInt (List.length model.documents)) ]
              else
                text ""
            ]

        -- Document list
        , div
            [ style "flex" "1"
            , style "padding" "0 8px"
            , style "overflow-y" "auto"
            ]
            [ if model.loading then
                div [ style "padding" "16px 6px", style "color" "#2a2a2a", style "font-size" "12px" ]
                    [ text "Loading…" ]
              else if List.isEmpty model.documents then
                sidebarEmptyState (toMsg OpenFilePicker)
              else
                div [ style "display" "flex", style "flex-direction" "column", style "gap" "2px", style "padding-bottom" "8px" ] <|
                    List.map
                        (\doc ->
                            Html.map toMsg
                                (DocumentCard.view doc NavigateToChat DeleteDocument False)
                        )
                        model.documents
            ]

        -- Ingesting indicator
        , if model.ingesting then
            div
                [ style "padding" "8px 14px"
                , style "display" "flex"
                , style "align-items" "center"
                , style "gap" "8px"
                , style "border-top" "1px solid #1a1a1a"
                ]
                [ div
                    [ style "width" "6px"
                    , style "height" "6px"
                    , style "border-radius" "50%"
                    , style "background" "#2563eb"
                    , class "anim-pulse"
                    , style "flex-shrink" "0"
                    ]
                    []
                , span [ style "font-size" "11px", style "color" "#555" ] [ text "Processing contract…" ]
                ]
          else
            text ""

        -- Status bar
        , div
            [ style "padding" "6px 10px 4px"
            , style "border-top" "1px solid #1a1a1a"
            ]
            [ case model.apiKeys of
                Just status ->
                    div
                        [ style "display" "flex"
                        , style "align-items" "center"
                        , style "gap" "5px"
                        , style "padding" "4px 4px"
                        ]
                        [ div
                            [ style "width" "5px"
                            , style "height" "5px"
                            , style "border-radius" "50%"
                            , style "flex-shrink" "0"
                            , style "background" (if status.ready then "#22c55e" else "#f59e0b")
                            ]
                            []
                        , div
                            [ style "font-size" "10px"
                            , style "color" (if status.ready then "#3a3a3a" else "#555")
                            , style "white-space" "nowrap"
                            , style "overflow" "hidden"
                            , style "text-overflow" "ellipsis"
                            ]
                            [ text
                                (if status.ready then
                                    shortModelName status.activeModel
                                 else
                                    "No API key — see Settings"
                                )
                            ]
                        ]

                Nothing ->
                    text ""
            ]

        -- Settings button
        , div [ style "padding" "0 10px 6px" ]
            [ button
                [ onClick onSettings
                , style "width" "100%"
                , style "background" (if settingsActive then "#1e1e1e" else "transparent")
                , style "border" (if settingsActive then "1px solid #2a2a2a" else "1px solid transparent")
                , style "border-radius" "7px"
                , style "padding" "8px 12px"
                , style "display" "flex"
                , style "align-items" "center"
                , style "gap" "8px"
                , style "cursor" "pointer"
                ]
                [ span [ style "font-size" "13px", style "color" (if settingsActive then "#888" else "#444") ] [ text "⚙" ]
                , span
                    [ style "font-size" "12px"
                    , style "font-weight" "600"
                    , style "color" (if settingsActive then "#aaa" else "#444")
                    ]
                    [ text "Settings" ]
                ]
            ]

        -- Open button
        , div [ style "padding" "0 10px 14px" ]
            [ button
                [ onClick (toMsg OpenFilePicker)
                , style "width" "100%"
                , style "background" "#1a2a4a"
                , style "border" "1px solid #2563eb"
                , style "border-radius" "8px"
                , style "padding" "9px 12px"
                , style "text-align" "center"
                , style "cursor" "pointer"
                , style "transition" "background 0.15s ease"
                ]
                [ span
                    [ style "color" "#60a5fa"
                    , style "font-size" "12px"
                    , style "font-weight" "700"
                    ]
                    [ text "+ Open Contract" ]
                ]
            ]
        ]


sidebarEmptyState : msg -> Html msg
sidebarEmptyState onOpen =
    div
        [ style "padding" "20px 6px"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "align-items" "center"
        , style "gap" "12px"
        , style "text-align" "center"
        ]
        [ div
            [ style "width" "36px"
            , style "height" "36px"
            , style "background" "#181818"
            , style "border" "1px dashed #2a2a2a"
            , style "border-radius" "8px"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "center"
            ]
            [ span [ style "font-size" "16px", style "color" "#2a2a2a" ] [ text "📄" ] ]
        , div [ style "font-size" "11px", style "color" "#2a2a2a", style "line-height" "1.5" ]
            [ text "No contracts yet" ]
        ]


shortModelName : String -> String
shortModelName model =
    -- "anthropic/claude-sonnet-4" → "claude-sonnet-4"
    -- "openai/gpt-4o" → "gpt-4o"
    case String.split "/" model of
        _ :: rest ->
            String.join "/" rest

        [] ->
            model
