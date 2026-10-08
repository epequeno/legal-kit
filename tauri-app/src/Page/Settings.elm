module Page.Settings exposing (Model, Msg(..), init, update, view)

import Html exposing (Html, button, div, input, label, span, text)
import Html.Attributes exposing (placeholder, style, type_, value)
import Html.Events exposing (onClick, onInput)
import Json.Decode as Decode
import Json.Encode as Encode
import Ports
import Types


type alias Model =
    { openrouterKey : String
    , openrouterModel : String
    , openaiKey : String
    , anthropicKey : String
    , anthropicModel : String
    , showDirect : Bool
    , saved : Bool
    , error : Maybe String
    -- Track which fields had values on load (for placeholder text)
    , hadOpenrouterKey : Bool
    , hadOpenaiKey : Bool
    , hadAnthropicKey : Bool
    , localEmbeddings : Bool
    , localEntailment : Bool
    , localReranker : Bool
    }


type Msg
    = OpenrouterKeyChanged String
    | OpenrouterModelChanged String
    | OpenaiKeyChanged String
    | AnthropicKeyChanged String
    | AnthropicModelChanged String
    | ToggleDirect
    | Save
    | SaveDone
    | GotSettings Decode.Value
    | GotApiStatus Decode.Value
    | ClearAllData
    | ClearAllDone
    | GotError String
    | NavigateHome


init : ( Model, Cmd Msg )
init =
    ( { openrouterKey = ""
      , openrouterModel = ""
      , openaiKey = ""
      , anthropicKey = ""
      , anthropicModel = ""
      , showDirect = False
      , saved = False
      , error = Nothing
      , hadOpenrouterKey = False
      , hadOpenaiKey = False
      , hadAnthropicKey = False
      , localEmbeddings = False
      , localEntailment = False
      , localReranker = False
      }
    , Cmd.batch
        [ Ports.invokeGetSettings ()
        , Ports.invokeGetApiKeyStatus ()
        ]
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        OpenrouterKeyChanged v ->
            ( { model | openrouterKey = v, saved = False }, Cmd.none )

        OpenrouterModelChanged v ->
            ( { model | openrouterModel = v, saved = False }, Cmd.none )

        OpenaiKeyChanged v ->
            ( { model | openaiKey = v, saved = False }, Cmd.none )

        AnthropicKeyChanged v ->
            ( { model | anthropicKey = v, saved = False }, Cmd.none )

        AnthropicModelChanged v ->
            ( { model | anthropicModel = v, saved = False }, Cmd.none )

        ToggleDirect ->
            ( { model | showDirect = not model.showDirect }, Cmd.none )

        Save ->
            ( model
            , Ports.invokeSaveSettings
                (Encode.object
                    [ ( "openrouter_key", Encode.string model.openrouterKey )
                    , ( "openrouter_model", Encode.string model.openrouterModel )
                    , ( "openai_key", Encode.string model.openaiKey )
                    , ( "anthropic_key", Encode.string model.anthropicKey )
                    , ( "anthropic_model", Encode.string model.anthropicModel )
                    ]
                )
            )

        SaveDone ->
            ( { model | saved = True, error = Nothing }, Cmd.none )

        GotApiStatus val ->
            case Decode.decodeValue Types.decodeApiKeyStatus val of
                Ok status ->
                    ( { model
                        | localEmbeddings = status.localEmbeddings
                        , localEntailment = status.localEntailment
                        , localReranker = status.localReranker
                      }
                    , Cmd.none
                    )

                Err _ ->
                    ( model, Cmd.none )

        GotSettings val ->
            let
                str field =
                    Decode.decodeValue (Decode.field field Decode.string) val
                        |> Result.withDefault ""
            in
            ( { model
                | openrouterModel = str "openrouter_model"
                , anthropicModel = str "anthropic_model"
                , hadOpenrouterKey = str "openrouter_key" == "set"
                , hadOpenaiKey = str "openai_key" == "set"
                , hadAnthropicKey = str "anthropic_key" == "set"
              }
            , Cmd.none
            )

        ClearAllData ->
            ( model, Ports.invokeClearAllData () )

        ClearAllDone ->
            ( { model
                | openrouterKey = ""
                , openrouterModel = ""
                , openaiKey = ""
                , anthropicKey = ""
                , anthropicModel = ""
                , hadOpenrouterKey = False
                , hadOpenaiKey = False
                , hadAnthropicKey = False
                , saved = False
              }
            , Cmd.none
            )

        GotError err ->
            ( { model | error = Just err }, Cmd.none )

        NavigateHome ->
            ( model, Cmd.none )


-- ── View ──────────────────────────────────────────────────────────────────


view : Model -> (Msg -> msg) -> Html msg
view model toMsg =
    div
        [ style "flex" "1"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "overflow" "hidden"
        ]
        [ -- Header
          div
            [ style "height" "44px"
            , style "border-bottom" "1px solid #222"
            , style "display" "flex"
            , style "align-items" "center"
            , style "padding" "0 32px"
            , style "flex-shrink" "0"
            ]
            [ span
                [ style "font-size" "13px"
                , style "font-weight" "600"
                , style "color" "#d4d4d4"
                ]
                [ text "Settings" ]
            ]

        -- Content
        , div
            [ style "flex" "1"
            , style "overflow-y" "auto"
            , style "padding" "32px"
            ]
            [ div
                [ style "max-width" "560px"
                , style "display" "flex"
                , style "flex-direction" "column"
                , style "gap" "32px"
                ]
                [ -- Error
                  case model.error of
                    Just err ->
                        div
                            [ style "background" "#2d0a0a"
                            , style "border" "1px solid #7f1d1d"
                            , style "border-radius" "8px"
                            , style "padding" "10px 14px"
                            , style "font-size" "12px"
                            , style "color" "#fca5a5"
                            ]
                            [ text err ]

                    Nothing ->
                        text ""

                -- API Keys section
                , div []
                    [ sectionLabel "API Keys"
                    , openrouterCard model toMsg
                    , directKeysCard model toMsg
                    ]

                -- Pipeline status
                , div []
                    [ sectionLabel "Pipeline Status"
                    , pipelineStatusCard model
                    ]

                -- Local Models section
                , div []
                    [ sectionLabel "Local Models"
                    , localModelsCard model.localEmbeddings model.localEntailment model.localReranker
                    ]

                -- Save button
                , div
                    [ style "display" "flex"
                    , style "align-items" "center"
                    , style "gap" "12px"
                    ]
                    [ button
                        [ onClick (toMsg Save)
                        , style "background" "#2563eb"
                        , style "border" "none"
                        , style "border-radius" "7px"
                        , style "padding" "10px 24px"
                        , style "cursor" "pointer"
                        ]
                        [ span
                            [ style "color" "#fff"
                            , style "font-size" "13px"
                            , style "font-weight" "600"
                            ]
                            [ text "Save Settings" ]
                        ]
                    , if model.saved then
                        span
                            [ style "font-size" "12px"
                            , style "color" "#22c55e"
                            , style "font-weight" "500"
                            ]
                            [ text "Saved" ]
                      else
                        span
                            [ style "font-size" "12px"
                            , style "color" "#3a3a3a"
                            ]
                            [ text "Changes apply immediately after save" ]
                    ]

                -- Danger zone
                , div []
                    [ sectionLabel "Danger Zone"
                    , div
                        [ style "background" "#1a1a1a"
                        , style "border" "1px solid #3f1515"
                        , style "border-radius" "10px"
                        , style "padding" "16px"
                        , style "display" "flex"
                        , style "align-items" "center"
                        , style "justify-content" "space-between"
                        ]
                        [ div []
                            [ div
                                [ style "font-size" "13px"
                                , style "font-weight" "600"
                                , style "color" "#fca5a5"
                                ]
                                [ text "Clear all data" ]
                            , div
                                [ style "font-size" "11px"
                                , style "color" "#555"
                                , style "margin-top" "2px"
                                ]
                                [ text "Delete all documents, chat history, and settings" ]
                            ]
                        , button
                            [ onClick (toMsg ClearAllData)
                            , style "background" "#7f1d1d"
                            , style "border" "none"
                            , style "border-radius" "6px"
                            , style "padding" "8px 16px"
                            , style "cursor" "pointer"
                            , style "flex-shrink" "0"
                            ]
                            [ span
                                [ style "color" "#fca5a5"
                                , style "font-size" "12px"
                                , style "font-weight" "600"
                                ]
                                [ text "Delete Everything" ]
                            ]
                        ]
                    ]
                ]
            ]
        ]


-- ── Components ────────────────────────────────────────────────────────────


sectionLabel : String -> Html msg
sectionLabel title =
    div
        [ style "font-size" "11px"
        , style "font-weight" "700"
        , style "color" "#444"
        , style "letter-spacing" "0.08em"
        , style "text-transform" "uppercase"
        , style "margin-bottom" "12px"
        ]
        [ text title ]


openrouterCard : Model -> (Msg -> msg) -> Html msg
openrouterCard model toMsg =
    div
        [ style "background" "#1a1a1a"
        , style "border" "1px solid #2a2a2a"
        , style "border-radius" "10px"
        , style "overflow" "hidden"
        , style "margin-bottom" "10px"
        ]
        [ -- Header
          div
            [ style "padding" "14px 16px"
            , style "border-bottom" "1px solid #222"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "space-between"
            ]
            [ div []
                [ div
                    [ style "font-size" "13px"
                    , style "font-weight" "600"
                    , style "color" "#d4d4d4"
                    ]
                    [ text "OpenRouter" ]
                , div
                    [ style "font-size" "11px"
                    , style "color" "#555"
                    , style "margin-top" "2px"
                    ]
                    [ text "Single key for all models — recommended" ]
                ]
            , if model.hadOpenrouterKey || not (String.isEmpty model.openrouterKey) then
                statusBadge "#22c55e" "Connected"
              else
                statusBadge "#555" "Not set"
            ]

        -- Fields
        , div
            [ style "padding" "14px 16px"
            , style "display" "flex"
            , style "flex-direction" "column"
            , style "gap" "12px"
            ]
            [ fieldInput "API Key" "password"
                (if model.hadOpenrouterKey && String.isEmpty model.openrouterKey then
                    "Already set — enter new key to change"
                 else
                    "sk-or-v1-..."
                )
                model.openrouterKey
                (toMsg << OpenrouterKeyChanged)
            , fieldInput "Model" "text"
                "anthropic/claude-sonnet-4"
                model.openrouterModel
                (toMsg << OpenrouterModelChanged)
            , div
                [ style "font-size" "11px"
                , style "color" "#3a3a3a"
                ]
                [ text "Any model from openrouter.ai/models" ]
            ]
        ]


directKeysCard : Model -> (Msg -> msg) -> Html msg
directKeysCard model toMsg =
    div
        [ style "background" "#1a1a1a"
        , style "border" "1px solid #2a2a2a"
        , style "border-radius" "10px"
        , style "overflow" "hidden"
        ]
        [ -- Toggle header
          div
            [ onClick (toMsg ToggleDirect)
            , style "padding" "14px 16px"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "space-between"
            , style "cursor" "pointer"
            ]
            [ div []
                [ div
                    [ style "font-size" "13px"
                    , style "font-weight" "600"
                    , style "color" "#555"
                    ]
                    [ text "Direct API Keys" ]
                , div
                    [ style "font-size" "11px"
                    , style "color" "#3a3a3a"
                    , style "margin-top" "2px"
                    ]
                    [ text "Use separate OpenAI + Anthropic keys" ]
                ]
            , span
                [ style "color" "#444"
                , style "font-size" "12px"
                ]
                [ text (if model.showDirect then "▾" else "▸") ]
            ]

        -- Expanded fields
        , if model.showDirect then
            div
                [ style "padding" "0 16px 14px"
                , style "border-top" "1px solid #222"
                , style "display" "flex"
                , style "flex-direction" "column"
                , style "gap" "12px"
                , style "padding-top" "14px"
                ]
                [ fieldInput "OpenAI API Key" "password"
                    (if model.hadOpenaiKey && String.isEmpty model.openaiKey then
                        "Already set — enter new key to change"
                     else
                        "sk-..."
                    )
                    model.openaiKey
                    (toMsg << OpenaiKeyChanged)
                , fieldInput "Anthropic API Key" "password"
                    (if model.hadAnthropicKey && String.isEmpty model.anthropicKey then
                        "Already set — enter new key to change"
                     else
                        "sk-ant-..."
                    )
                    model.anthropicKey
                    (toMsg << AnthropicKeyChanged)
                , fieldInput "Anthropic Model" "text"
                    "claude-sonnet-4-20250514"
                    model.anthropicModel
                    (toMsg << AnthropicModelChanged)
                ]

          else
            text ""
        ]


pipelineStatusCard : Model -> Html msg
pipelineStatusCard model =
    let
        apiReady =
            model.hadOpenrouterKey || model.hadOpenaiKey || model.hadAnthropicKey

        capabilities =
            [ ( "API Generation", apiReady, "LLM answers via API" )
            , ( "Local Embeddings", model.localEmbeddings, "Offline semantic search" )
            , ( "Citation Verifier", model.localEntailment, "Entailment scoring (DeBERTa)" )
            , ( "Legal Reranker", model.localReranker, "Cross-encoder reranking (XLM-R)" )
            ]

        activeCount =
            List.length (List.filter (\( _, on, _ ) -> on) capabilities)
    in
    div
        [ style "background" "#161616"
        , style "border" "1px solid #222"
        , style "border-radius" "10px"
        , style "overflow" "hidden"
        ]
        [ -- Summary header
          div
            [ style "padding" "12px 16px"
            , style "border-bottom" "1px solid #1e1e1e"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "space-between"
            ]
            [ span [ style "font-size" "12px", style "color" "#555", style "font-weight" "500" ]
                [ text "Active pipeline" ]
            , div
                [ style "display" "flex"
                , style "align-items" "center"
                , style "gap" "6px"
                ]
                [ div
                    [ style "width" "6px"
                    , style "height" "6px"
                    , style "border-radius" "50%"
                    , style "background" (if activeCount >= 2 then "#22c55e" else if activeCount >= 1 then "#f59e0b" else "#555")
                    ]
                    []
                , span
                    [ style "font-size" "11px"
                    , style "font-weight" "700"
                    , style "color" (if activeCount >= 2 then "#22c55e" else if activeCount >= 1 then "#f59e0b" else "#555")
                    ]
                    [ text (String.fromInt activeCount ++ "/" ++ String.fromInt (List.length capabilities) ++ " capabilities active") ]
                ]
            ]

        -- Capability rows
        , div []
            (List.indexedMap
                (\i ( cap, on, desc ) ->
                    div
                        [ style "padding" "11px 16px"
                        , style "display" "flex"
                        , style "align-items" "center"
                        , style "justify-content" "space-between"
                        , style "gap" "12px"
                        , if i < List.length capabilities - 1 then
                            style "border-bottom" "1px solid #1a1a1a"
                          else
                            style "" ""
                        ]
                        [ div []
                            [ div
                                [ style "font-size" "12px"
                                , style "font-weight" "600"
                                , style "color" (if on then "#aaa" else "#333")
                                ]
                                [ text cap ]
                            , div
                                [ style "font-size" "10px"
                                , style "color" "#2a2a2a"
                                , style "margin-top" "2px"
                                ]
                                [ text desc ]
                            ]
                        , if on then
                            statusBadge "#22c55e" "Active"
                          else
                            statusBadge "#333" "Inactive"
                        ]
                )
                capabilities
            )
        ]


localModelsCard : Bool -> Bool -> Bool -> Html msg
localModelsCard embeddingsLoaded entailmentLoaded rerankerLoaded =
    div
        [ style "background" "#161616"
        , style "border" "1px solid #222"
        , style "border-radius" "10px"
        , style "overflow" "hidden"
        ]
        [ div
            [ style "padding" "10px 16px"
            , style "border-bottom" "1px solid #1e1e1e"
            ]
            [ div [ style "font-size" "11px", style "color" "#2a2a2a", style "line-height" "1.5" ]
                [ text "Models are loaded automatically from "
                , span [ style "font-family" "monospace", style "color" "#333" ] [ text "models/" ]
                , text " or "
                , span [ style "font-family" "monospace", style "color" "#333" ] [ text "~/.legalkit/models/" ]
                , text " on startup."
                ]
            ]
        , modelRow "Embeddings" "bge-base-en-v1.5" "Apple Accelerate CPU" embeddingsLoaded False
        , modelRow "Citation Verifier" "DeBERTa-v3-large" "Apple Accelerate CPU" entailmentLoaded False
        , modelRow "Legal Reranker" "XLM-RoBERTa-large" "Apple Accelerate CPU" rerankerLoaded True
        ]


modelRow : String -> String -> String -> Bool -> Bool -> Html msg
modelRow title modelName device isLoaded isLast =
    div
        [ style "padding" "12px 16px"
        , style "display" "flex"
        , style "align-items" "center"
        , style "justify-content" "space-between"
        , style "gap" "12px"
        , if not isLast then
            style "border-bottom" "1px solid #1a1a1a"
          else
            style "" ""
        ]
        [ div [ style "flex" "1" ]
            [ div
                [ style "font-size" "12px"
                , style "font-weight" "600"
                , style "color" (if isLoaded then "#aaa" else "#333")
                ]
                [ text title ]
            , div
                [ style "display" "flex"
                , style "gap" "8px"
                , style "margin-top" "3px"
                , style "align-items" "center"
                ]
                [ span
                    [ style "font-size" "10px"
                    , style "color" "#2a2a2a"
                    , style "font-family" "monospace"
                    ]
                    [ text modelName ]
                , if isLoaded then
                    span
                        [ style "font-size" "10px"
                        , style "color" "#1d4030"
                        , style "background" "#0a1f14"
                        , style "border" "1px solid #14532d"
                        , style "border-radius" "3px"
                        , style "padding" "1px 5px"
                        ]
                        [ text device ]
                  else
                    text ""
                ]
            ]
        , if isLoaded then
            statusBadge "#22c55e" "Loaded"
          else
            statusBadge "#2a2a2a" "Not found"
        ]


fieldInput : String -> String -> String -> String -> (String -> msg) -> Html msg
fieldInput labelText inputType placeholderText val onchange =
    div []
        [ label
            [ style "font-size" "11px"
            , style "font-weight" "600"
            , style "color" "#555"
            , style "display" "block"
            , style "margin-bottom" "6px"
            ]
            [ text labelText ]
        , input
            [ type_ inputType
            , placeholder placeholderText
            , value val
            , onInput onchange
            , style "width" "100%"
            , style "background" "#111"
            , style "border" "1px solid #2a2a2a"
            , style "border-radius" "6px"
            , style "padding" "8px 12px"
            , style "color" "#d4d4d4"
            , style "font-size" "13px"
            , style "font-family" "monospace"
            , style "box-sizing" "border-box"
            , style "outline" "none"
            ]
            []
        ]


statusBadge : String -> String -> Html msg
statusBadge color label =
    div
        [ style "display" "flex"
        , style "align-items" "center"
        , style "gap" "6px"
        , style "flex-shrink" "0"
        ]
        [ div
            [ style "width" "6px"
            , style "height" "6px"
            , style "border-radius" "50%"
            , style "background" color
            ]
            []
        , span
            [ style "font-size" "11px"
            , style "color" color
            , style "font-weight" "600"
            ]
            [ text label ]
        ]
