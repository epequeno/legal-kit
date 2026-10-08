port module Page.Chat exposing (Model, Msg(..), init, update, view, askEvent)

import Api
import Components.CitationCard as CitationCard
import Dict exposing (Dict)
import Html exposing (Html, button, div, h2, input, text, textarea)
import Html.Attributes exposing (disabled, placeholder, style, value)
import Html.Events exposing (onClick, onInput)
import Http
import Json.Decode as Decode
import Types exposing (AskResult, Document)


type alias Model =
    { docId : String
    , document : Maybe Document
    , messages : List Message
    , questionInput : String
    , loading : Bool
    , streamingProgress : Maybe String
    , expandedClaims : Dict Int Bool
    , error : Maybe String
    }


type Message
    = UserMessage String
    | AssistantMessage AskResult


type Msg
    = GotDocument (Result Http.Error Document)
    | QuestionInputChanged String
    | SubmitQuestion
    | AskEvent Decode.Value
    | ToggleClaim Int
    | NavigateBack


init : String -> ( Model, Cmd Msg )
init docId =
    ( { docId = docId
      , document = Nothing
      , messages = []
      , questionInput = ""
      , loading = True
      , streamingProgress = Nothing
      , expandedClaims = Dict.empty
      , error = Nothing
      }
    , Api.getDocument docId GotDocument
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        GotDocument result ->
            case result of
                Ok doc ->
                    ( { model | document = Just doc, loading = False }, Cmd.none )

                Err _ ->
                    ( { model | error = Just "Failed to load document", loading = False }, Cmd.none )

        QuestionInputChanged value ->
            ( { model | questionInput = value }, Cmd.none )

        SubmitQuestion ->
            if String.isEmpty (String.trim model.questionInput) then
                ( model, Cmd.none )

            else
                let
                    question =
                        String.trim model.questionInput

                    newMessages =
                        model.messages ++ [ UserMessage question ]
                in
                ( { model
                    | messages = newMessages
                    , questionInput = ""
                    , streamingProgress = Just "Starting..."
                    , loading = True
                  }
                , startAsk { docId = model.docId, question = question, topK = 5 }
                )

        AskEvent value ->
            handleAskEvent value model

        ToggleClaim index ->
            ( { model
                | expandedClaims =
                    Dict.update index
                        (\current ->
                            case current of
                                Just expanded ->
                                    Just (not expanded)

                                Nothing ->
                                    Just True
                        )
                        model.expandedClaims
              }
            , Cmd.none
            )

        NavigateBack ->
            -- Handled by Main.elm
            ( model, Cmd.none )


handleAskEvent : Decode.Value -> Model -> ( Model, Cmd Msg )
handleAskEvent value model =
    case Decode.decodeValue askEventDecoder value of
        Ok event ->
            case event of
                ProgressEvent progress ->
                    ( { model | streamingProgress = Just progress }, Cmd.none )

                ResultEvent result ->
                    ( { model
                        | messages = model.messages ++ [ AssistantMessage result ]
                        , streamingProgress = Nothing
                        , loading = False
                        , expandedClaims = Dict.empty
                      }
                    , Cmd.none
                    )

                ErrorEvent err ->
                    ( { model
                        | error = Just err
                        , streamingProgress = Nothing
                        , loading = False
                      }
                    , Cmd.none
                    )

        Err _ ->
            ( { model | error = Just "Invalid event received", streamingProgress = Nothing, loading = False }, Cmd.none )


view : Model -> (Msg -> msg) -> (msg -> msg) -> Html msg
view model toMsg onBack =
    div
        [ style "max-width" "960px"
        , style "margin" "0 auto"
        , style "padding" "20px"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "height" "100vh"
        ]
        [ -- Header
          div
            [ style "display" "flex"
            , style "justify-content" "space-between"
            , style "align-items" "center"
            , style "margin-bottom" "20px"
            , style "padding-bottom" "16px"
            , style "border-bottom" "1px solid #e5e7eb"
            ]
            [ div []
                [ h2
                    [ style "font-size" "24px"
                    , style "font-weight" "700"
                    , style "color" "#111827"
                    , style "margin" "0"
                    ]
                    [ text
                        (case model.document of
                            Just doc ->
                                doc.filename

                            Nothing ->
                                "Loading..."
                        )
                    ]
                ]
            , button
                [ onClick (toMsg NavigateBack)
                , style "background" "#6b7280"
                , style "color" "white"
                , style "border" "none"
                , style "border-radius" "6px"
                , style "padding" "8px 16px"
                , style "font-size" "14px"
                , style "cursor" "pointer"
                ]
                [ text "← Back" ]
            ]
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
        , -- Messages area
          div
            [ style "flex" "1"
            , style "overflow-y" "auto"
            , style "margin-bottom" "20px"
            ]
            (if List.isEmpty model.messages then
                [ div
                    [ style "text-align" "center"
                    , style "padding" "60px 20px"
                    , style "color" "#6b7280"
                    ]
                    [ div [ style "font-size" "48px", style "margin-bottom" "16px" ]
                        [ text "💬" ]
                    , div [ style "font-size" "18px", style "font-weight" "500" ]
                        [ text "Ask a question about this document" ]
                    ]
                ]

             else
                List.indexedMap (\i m -> Html.map toMsg (viewMessage m i model.expandedClaims)) model.messages
                    ++ (case model.streamingProgress of
                            Just progress ->
                                [ viewProgressIndicator progress ]

                            Nothing ->
                                []
                       )
            )
        , -- Input area
          div
            [ style "border-top" "1px solid #e5e7eb"
            , style "padding-top" "16px"
            ]
            [ div [ style "display" "flex", style "gap" "12px" ]
                [ input
                    [ value model.questionInput
                    , onInput (toMsg << QuestionInputChanged)
                    , placeholder "Ask a question..."
                    , disabled model.loading
                    , style "flex" "1"
                    , style "padding" "12px"
                    , style "border" "1px solid #d1d5db"
                    , style "border-radius" "6px"
                    , style "font-size" "14px"
                    , style "outline" "none"
                    ]
                    []
                , button
                    [ onClick (toMsg SubmitQuestion)
                    , disabled (model.loading || String.isEmpty (String.trim model.questionInput))
                    , style "background"
                        (if model.loading || String.isEmpty (String.trim model.questionInput) then
                            "#9ca3af"

                         else
                            "#2563eb"
                        )
                    , style "color" "white"
                    , style "border" "none"
                    , style "border-radius" "6px"
                    , style "padding" "12px 24px"
                    , style "font-size" "14px"
                    , style "font-weight" "500"
                    , style "cursor"
                        (if model.loading then
                            "not-allowed"

                         else
                            "pointer"
                        )
                    ]
                    [ text
                        (if model.loading then
                            "Asking..."

                         else
                            "Ask"
                        )
                    ]
                ]
            ]
        ]


viewMessage : Message -> Int -> Dict Int Bool -> Html Msg
viewMessage message msgIndex expandedClaims =
    case message of
        UserMessage question ->
            div
                [ style "background" "#e0f2fe"
                , style "padding" "16px"
                , style "border-radius" "8px"
                , style "margin-bottom" "16px"
                , style "margin-left" "20%"
                ]
                [ div
                    [ style "font-size" "12px"
                    , style "font-weight" "600"
                    , style "color" "#0c4a6e"
                    , style "margin-bottom" "8px"
                    ]
                    [ text "You" ]
                , div
                    [ style "font-size" "14px"
                    , style "color" "#0c4a6e"
                    , style "line-height" "1.5"
                    ]
                    [ text question ]
                ]

        AssistantMessage result ->
            div
                [ style "background" "white"
                , style "border" "1px solid #e5e7eb"
                , style "padding" "16px"
                , style "border-radius" "8px"
                , style "margin-bottom" "16px"
                , style "margin-right" "20%"
                ]
                [ div
                    [ style "font-size" "12px"
                    , style "font-weight" "600"
                    , style "color" "#2563eb"
                    , style "margin-bottom" "8px"
                    ]
                    [ text "Assistant" ]
                , div
                    [ style "font-size" "14px"
                    , style "color" "#374151"
                    , style "line-height" "1.6"
                    , style "margin-bottom" "16px"
                    ]
                    [ text result.answer ]
                , div
                    [ style "border-top" "1px solid #e5e7eb"
                    , style "padding-top" "16px"
                    , style "margin-top" "16px"
                    ]
                    [ div
                        [ style "font-size" "13px"
                        , style "font-weight" "600"
                        , style "color" "#6b7280"
                        , style "margin-bottom" "12px"
                        ]
                        [ text "Evidence & Citations" ]
                    , div []
                        (List.indexedMap
                            (\claimIdx claim ->
                                let
                                    globalIdx =
                                        msgIndex * 1000 + claimIdx

                                    isExpanded =
                                        Dict.get globalIdx expandedClaims |> Maybe.withDefault False
                                in
                                CitationCard.view claim isExpanded (ToggleClaim globalIdx)
                            )
                            result.claims
                        )
                    , if not (List.isEmpty result.gaps) then
                        div
                            [ style "margin-top" "12px"
                            , style "padding" "12px"
                            , style "background" "#fef3c7"
                            , style "border-radius" "6px"
                            ]
                            [ div
                                [ style "font-size" "12px"
                                , style "font-weight" "600"
                                , style "color" "#78350f"
                                , style "margin-bottom" "8px"
                                ]
                                [ text "⚠️ Knowledge Gaps" ]
                            , div []
                                (List.map
                                    (\gap ->
                                        div
                                            [ style "font-size" "12px"
                                            , style "color" "#78350f"
                                            , style "margin-bottom" "4px"
                                            ]
                                            [ text ("• " ++ gap) ]
                                    )
                                    result.gaps
                                )
                            ]

                      else
                        text ""
                    , div
                        [ style "margin-top" "12px"
                        , style "font-size" "12px"
                        , style "color" "#9ca3af"
                        ]
                        [ text ("Overall confidence: " ++ result.overallConfidence.level ++ " • Time: " ++ String.fromFloat result.elapsedSeconds ++ "s") ]
                    ]
                ]


viewProgressIndicator : String -> Html msg
viewProgressIndicator progress =
    div
        [ style "background" "#f3f4f6"
        , style "padding" "16px"
        , style "border-radius" "8px"
        , style "margin-bottom" "16px"
        , style "text-align" "center"
        ]
        [ div
            [ style "font-size" "14px"
            , style "color" "#6b7280"
            , style "font-style" "italic"
            ]
            [ text progress ]
        ]



-- PORTS


port startAsk : { docId : String, question : String, topK : Int } -> Cmd msg


port askEvent : (Decode.Value -> msg) -> Sub msg



-- DECODERS


type AskEvent
    = ProgressEvent String
    | ResultEvent AskResult
    | ErrorEvent String


askEventDecoder : Decode.Decoder AskEvent
askEventDecoder =
    Decode.field "event" Decode.string
        |> Decode.andThen
            (\eventType ->
                case eventType of
                    "progress" ->
                        Decode.map ProgressEvent (Decode.field "message" Decode.string)

                    "result" ->
                        Decode.map ResultEvent (Decode.field "data" Types.decodeAskResult)

                    "error" ->
                        Decode.map ErrorEvent (Decode.field "message" Decode.string)

                    _ ->
                        Decode.fail ("Unknown event type: " ++ eventType)
            )
