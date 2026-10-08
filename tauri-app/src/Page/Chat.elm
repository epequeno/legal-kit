module Page.Chat exposing (Model, Msg(..), init, update, view, viewSources)

import Html exposing (Html, button, div, span, text, textarea)
import Html.Attributes exposing (class, disabled, id, placeholder, style, value)
import Html.Events exposing (custom, keyCode, on, onClick, onInput)
import Json.Decode as JD
import Json.Decode as Decode
import Json.Encode as Encode
import Markdown
import Ports
import Process
import Set exposing (Set)
import Task
import Types exposing (AnswerReport, Citation, Document, DocumentOutline, OutlineSection, decodeAnswerReport, decodeAnswerReportList, decodeDocumentOutline)


type alias Model =
    { doc : Document
    , question : String
    , thinking : Bool
    , report : Maybe AnswerReport
    , history : List AnswerReport
    , error : Maybe String
    , activeCitationId : Maybe String  -- highlighted citation node id
    , copiedIdx : Maybe Int            -- index of answer with active "Copied!" state
    , expandedCitations : Set String   -- set of citation nodeIds with expanded quotes
    , outline : Maybe DocumentOutline  -- contract structure overview
    , outlineExpanded : Bool           -- whether the top-sections list is expanded
    }


type Msg
    = QuestionChanged String
    | SubmitQuestion
    | GotAnswer Decode.Value
    | GotHistoryResult Decode.Value
    | GotOutline Decode.Value
    | ClearHistory
    | GotError String
    | NavigateHome
    | CitationClicked String
    | CopyAnswer Int String
    | CopyClear
    | ToggleCitationExpand String
    | ToggleOutlineExpanded
    | UseSuggestedQuestion String


init : Document -> ( Model, Cmd Msg )
init doc =
    ( { doc = doc
      , question = ""
      , thinking = False
      , report = Nothing
      , history = []
      , error = Nothing
      , activeCitationId = Nothing
      , copiedIdx = Nothing
      , expandedCitations = Set.empty
      , outline = Nothing
      , outlineExpanded = False
      }
    , Cmd.batch
        [ Ports.invokeLoadChatHistory doc.id
        , Ports.invokeGetDocumentOutline doc.id
        ]
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        QuestionChanged q ->
            ( { model | question = q }, Cmd.none )

        SubmitQuestion ->
            if String.trim model.question == "" || model.thinking then
                ( model, Cmd.none )
            else
                ( { model | thinking = True, error = Nothing, question = "", activeCitationId = Nothing }
                , Ports.invokeAskQuestion
                    (Encode.object
                        [ ( "doc_id", Encode.string model.doc.id )
                        , ( "question", Encode.string model.question )
                        ]
                    )
                )

        GotAnswer val ->
            case Decode.decodeValue decodeAnswerReport val of
                Ok report ->
                    ( { model
                        | thinking = False
                        , report = Just report
                        , history = model.history ++ [ report ]
                      }
                    , Cmd.none
                    )

                Err e ->
                    ( { model
                        | thinking = False
                        , error = Just ("Failed to parse answer: " ++ Decode.errorToString e)
                      }
                    , Cmd.none
                    )

        GotHistoryResult val ->
            case Decode.decodeValue decodeAnswerReportList val of
                Ok reports ->
                    ( { model
                        | history = reports
                        , report = List.head (List.reverse reports)
                      }
                    , Cmd.none
                    )

                Err _ ->
                    ( model, Cmd.none )

        GotOutline val ->
            case Decode.decodeValue decodeDocumentOutline val of
                Ok outline ->
                    ( { model | outline = Just outline }, Cmd.none )

                Err _ ->
                    ( model, Cmd.none )

        ClearHistory ->
            ( { model | history = [], report = Nothing, activeCitationId = Nothing, copiedIdx = Nothing }
            , Ports.invokeClearChatHistory model.doc.id
            )

        GotError err ->
            ( { model | thinking = False, error = Just err }, Cmd.none )

        NavigateHome ->
            ( model, Cmd.none )

        CitationClicked nodeId ->
            let
                next =
                    if model.activeCitationId == Just nodeId then
                        Nothing
                    else
                        Just nodeId
            in
            ( { model | activeCitationId = next }, Cmd.none )

        CopyAnswer idx text ->
            ( { model | copiedIdx = Just idx }
            , Cmd.batch
                [ Ports.copyToClipboard text
                , Task.perform (\_ -> CopyClear) (Process.sleep 2000)
                ]
            )

        CopyClear ->
            ( { model | copiedIdx = Nothing }, Cmd.none )

        ToggleCitationExpand nodeId ->
            let
                next =
                    if Set.member nodeId model.expandedCitations then
                        Set.remove nodeId model.expandedCitations
                    else
                        Set.insert nodeId model.expandedCitations
            in
            ( { model | expandedCitations = next }, Cmd.none )

        ToggleOutlineExpanded ->
            ( { model | outlineExpanded = not model.outlineExpanded }, Cmd.none )

        UseSuggestedQuestion q ->
            ( { model | question = q }, Cmd.none )


-- ── Views ──────────────────────────────────────────────────────────────────


view : Model -> (Msg -> msg) -> msg -> Html msg
view model toMsg goHome =
    div
        [ style "flex" "1"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "height" "100%"
        , style "min-width" "0"
        ]
        [ -- Doc header bar
          div
            [ style "height" "44px"
            , style "border-bottom" "1px solid #222"
            , style "display" "flex"
            , style "align-items" "center"
            , style "padding" "0 16px"
            , style "gap" "12px"
            , style "flex-shrink" "0"
            ]
            [ button
                [ onClick goHome
                , style "background" "none"
                , style "border" "none"
                , style "color" "#444"
                , style "font-size" "16px"
                , style "cursor" "pointer"
                , style "padding" "0 4px 0 0"
                , style "line-height" "1"
                ]
                [ text "←" ]
            , div
                [ style "font-size" "13px"
                , style "font-weight" "600"
                , style "color" "#d4d4d4"
                , style "white-space" "nowrap"
                , style "overflow" "hidden"
                , style "text-overflow" "ellipsis"
                , style "flex" "1"
                ]
                [ text (stripExtension model.doc.filename) ]
            , span
                [ style "font-size" "11px"
                , style "color" "#333"
                , style "flex-shrink" "0"
                ]
                [ text (String.fromInt model.doc.nChunks ++ " sections") ]
            , if List.isEmpty model.history then
                text ""
              else
                button
                    [ onClick (toMsg ClearHistory)
                    , style "background" "none"
                    , style "border" "1px solid #262626"
                    , style "border-radius" "5px"
                    , style "color" "#444"
                    , style "font-size" "11px"
                    , style "padding" "4px 10px"
                    , style "cursor" "pointer"
                    , style "flex-shrink" "0"
                    ]
                    [ text "Clear" ]
            ]

        -- Conversation area
        , div
            [ style "flex" "1"
            , style "overflow-y" "auto"
            , style "padding" "24px 28px"
            , style "display" "flex"
            , style "flex-direction" "column"
            , style "gap" "24px"
            ]
            (if List.isEmpty model.history && not model.thinking then
                [ viewEmptyState model toMsg ]
             else
                List.concat
                    (List.indexedMap (viewQAPair toMsg model.activeCitationId model.copiedIdx) model.history)
                    ++ (if model.thinking then
                            [ thinkingIndicator (stripExtension model.doc.filename) ]
                        else
                            []
                       )
                    ++ (case model.error of
                            Just err ->
                                [ errorBubble err ]
                            Nothing ->
                                []
                       )
            )

        -- Input bar
        , div
            [ style "padding" "12px 16px 16px"
            , style "border-top" "1px solid #1e1e1e"
            , style "flex-shrink" "0"
            ]
            [ div
                [ style "display" "flex"
                , style "gap" "10px"
                , style "align-items" "flex-end"
                , style "background" "#181818"
                , style "border" "1px solid #2a2a2a"
                , style "border-radius" "12px"
                , style "padding" "10px 12px"
                ]
                [ textarea
                    [ value model.question
                    , onInput (toMsg << QuestionChanged)
                    , custom "keydown" (enterToSubmit (toMsg SubmitQuestion))
                    , placeholder "Ask anything about this contract…"
                    , style "flex" "1"
                    , style "background" "none"
                    , style "border" "none"
                    , style "outline" "none"
                    , style "color" "#d4d4d4"
                    , style "font-size" "14px"
                    , style "font-family" "inherit"
                    , style "resize" "none"
                    , style "min-height" "22px"
                    , style "max-height" "120px"
                    , style "line-height" "1.55"
                    , style "overflow-y" "auto"
                    ]
                    []
                , button
                    [ onClick (toMsg SubmitQuestion)
                    , disabled (model.thinking || String.trim model.question == "")
                    , style "background" "#2563eb"
                    , style "border" "none"
                    , style "border-radius" "8px"
                    , style "width" "36px"
                    , style "height" "36px"
                    , style "display" "flex"
                    , style "align-items" "center"
                    , style "justify-content" "center"
                    , style "cursor" (if model.thinking then "not-allowed" else "pointer")
                    , style "opacity" (if model.thinking || String.trim model.question == "" then "0.45" else "1")
                    , style "flex-shrink" "0"
                    , style "transition" "opacity 0.15s ease"
                    ]
                    [ span [ style "color" "#fff", style "font-size" "16px", style "line-height" "1" ]
                        [ text "↑" ]
                    ]
                ]
            , div
                [ style "display" "flex"
                , style "align-items" "center"
                , style "justify-content" "flex-end"
                , style "padding-top" "6px"
                ]
                [ span [ style "font-size" "10px", style "color" "#2a2a2a" ]
                    [ text "⏎ to send · Shift+⏎ for newline" ]
                ]
            ]
        ]


-- ── Empty state / Contract overview ───────────────────────────────────────


suggestedQuestions : List String
suggestedQuestions =
    [ "What are the termination conditions?"
    , "Does this contract include an exclusivity clause?"
    , "What are the payment terms and deadlines?"
    , "What IP does each party own?"
    , "Are there any non-compete or non-solicitation clauses?"
    , "What happens in case of a material breach?"
    ]


viewEmptyState : Model -> (Msg -> msg) -> Html msg
viewEmptyState model toMsg =
    div
        [ style "display" "flex"
        , style "flex-direction" "column"
        , style "gap" "20px"
        , style "padding" "28px 32px"
        , style "max-width" "640px"
        , style "width" "100%"
        , style "align-self" "center"
        , class "anim-fade-in"
        ]
        [ -- Contract overview card
          viewOutlineCard model toMsg

        -- Suggested questions
        , div []
            [ div
                [ style "font-size" "10px"
                , style "font-weight" "700"
                , style "color" "#2a2a2a"
                , style "letter-spacing" "0.08em"
                , style "text-transform" "uppercase"
                , style "margin-bottom" "10px"
                ]
                [ text "Suggested Questions" ]
            , div
                [ style "display" "flex"
                , style "flex-direction" "column"
                , style "gap" "6px"
                , class "anim-stagger"
                ]
                (List.map (viewSuggestion toMsg) suggestedQuestions)
            ]
        ]


viewOutlineCard : Model -> (Msg -> msg) -> Html msg
viewOutlineCard model toMsg =
    div
        [ style "background" "#161616"
        , style "border" "1px solid #222"
        , style "border-radius" "12px"
        , style "overflow" "hidden"
        ]
        [ -- Header row
          div
            [ style "padding" "14px 18px"
            , style "display" "flex"
            , style "align-items" "flex-start"
            , style "justify-content" "space-between"
            , style "gap" "16px"
            ]
            [ div [ style "flex" "1", style "min-width" "0" ]
                [ div
                    [ style "font-size" "15px"
                    , style "font-weight" "700"
                    , style "color" "#d4d4d4"
                    , style "white-space" "nowrap"
                    , style "overflow" "hidden"
                    , style "text-overflow" "ellipsis"
                    ]
                    [ text (stripExtension model.doc.filename) ]
                , div
                    [ style "display" "flex"
                    , style "gap" "16px"
                    , style "margin-top" "8px"
                    , style "flex-wrap" "wrap"
                    ]
                    (case model.outline of
                        Just outline ->
                            [ statPill (String.fromInt outline.totalSections) "sections"
                            , statPill (String.fromInt (List.length outline.topSections)) "top-level"
                            , if outline.crossRefsCount > 0 then
                                statPill (String.fromInt outline.crossRefsCount) "cross-refs"
                              else
                                text ""
                            , statPill (formatBytes model.doc.sizeBytes) ""
                            ]

                        Nothing ->
                            [ statPill (String.fromInt model.doc.nChunks) "sections"
                            , statPill (formatBytes model.doc.sizeBytes) ""
                            ]
                    )
                ]

            -- File type badge
            , div
                [ style "background" "#1e3a5f"
                , style "border" "1px solid #2563eb"
                , style "border-radius" "6px"
                , style "padding" "4px 10px"
                , style "flex-shrink" "0"
                ]
                [ span
                    [ style "font-size" "11px"
                    , style "font-weight" "800"
                    , style "color" "#60a5fa"
                    ]
                    [ text (fileTypeLabel model.doc.filename) ]
                ]
            ]

        -- Section outline (if loaded and has content)
        , case model.outline of
            Just outline ->
                if List.isEmpty outline.topSections then
                    text ""
                else
                    viewSectionOutline outline model.outlineExpanded toMsg

            Nothing ->
                text ""
        ]


viewSectionOutline : DocumentOutline -> Bool -> (Msg -> msg) -> Html msg
viewSectionOutline outline expanded toMsg =
    let
        visibleSections =
            if expanded then
                outline.topSections
            else
                List.take 6 outline.topSections

        hasMore =
            List.length outline.topSections > 6
    in
    div
        [ style "border-top" "1px solid #1e1e1e" ]
        [ -- Section list
          div
            [ style "padding" "10px 18px 4px"
            , style "display" "flex"
            , style "flex-direction" "column"
            , style "gap" "3px"
            ]
            (List.map viewSectionRow visibleSections)

        -- Show more / less toggle
        , if hasMore then
            button
                [ onClick (toMsg ToggleOutlineExpanded)
                , style "width" "100%"
                , style "background" "none"
                , style "border" "none"
                , style "border-top" "1px solid #1a1a1a"
                , style "padding" "8px 18px"
                , style "cursor" "pointer"
                , style "text-align" "left"
                , style "font-size" "11px"
                , style "font-weight" "600"
                , style "color" "#2a2a2a"
                ]
                [ text
                    (if expanded then
                        "Show less ↑"
                     else
                        "Show all " ++ String.fromInt (List.length outline.topSections) ++ " sections ↓"
                    )
                ]
          else
            div [ style "height" "10px" ] []
        ]


viewSectionRow : OutlineSection -> Html msg
viewSectionRow section =
    div
        [ style "display" "flex"
        , style "align-items" "center"
        , style "gap" "10px"
        , style "padding" "4px 0"
        ]
        [ -- Type indicator dot
          div
            [ style "width" "5px"
            , style "height" "5px"
            , style "border-radius" "50%"
            , style "flex-shrink" "0"
            , style "background" (sectionTypeColor section.nodeType)
            ]
            []
        , span
            [ style "font-size" "11px"
            , style "font-weight" "600"
            , style "color" "#3a3a3a"
            , style "font-family" "monospace"
            , style "flex-shrink" "0"
            , style "min-width" "60px"
            ]
            [ text section.id ]
        , span
            [ style "font-size" "11px"
            , style "color" "#333"
            , style "white-space" "nowrap"
            , style "overflow" "hidden"
            , style "text-overflow" "ellipsis"
            ]
            [ text section.title ]
        ]


sectionTypeColor : String -> String
sectionTypeColor nodeType =
    case nodeType of
        "article" -> "#3b82f6"
        "exhibit" -> "#a855f7"
        "schedule" -> "#a855f7"
        "preamble" -> "#f59e0b"
        _ -> "#374151"


statPill : String -> String -> Html msg
statPill value label =
    div
        [ style "display" "flex"
        , style "align-items" "baseline"
        , style "gap" "4px"
        ]
        [ span
            [ style "font-size" "13px"
            , style "font-weight" "700"
            , style "color" "#555"
            ]
            [ text value ]
        , if String.isEmpty label then
            text ""
          else
            span
                [ style "font-size" "10px"
                , style "color" "#2e2e2e"
                , style "font-weight" "500"
                ]
                [ text label ]
        ]


formatBytes : Int -> String
formatBytes bytes =
    if bytes < 1024 then
        String.fromInt bytes ++ " B"
    else if bytes < 1048576 then
        String.fromFloat (toFloat (round (toFloat bytes / 102.4)) / 10) ++ " KB"
    else
        String.fromFloat (toFloat (round (toFloat bytes / 104857.6)) / 10) ++ " MB"


fileTypeLabel : String -> String
fileTypeLabel filename =
    case String.toLower (String.right 4 filename) of
        ".pdf" -> "PDF"
        "docx" -> "DOCX"
        ".txt" -> "TXT"
        _ -> "DOC"


viewSuggestion : (Msg -> msg) -> String -> Html msg
viewSuggestion toMsg q =
    button
        [ onClick (toMsg (UseSuggestedQuestion q))
        , style "width" "100%"
        , style "text-align" "left"
        , style "background" "#161616"
        , style "border" "1px solid #1e1e1e"
        , style "border-radius" "8px"
        , style "padding" "10px 14px"
        , style "color" "#555"
        , style "font-size" "13px"
        , style "font-family" "inherit"
        , style "cursor" "pointer"
        , style "line-height" "1.4"
        , style "transition" "border-color 0.15s ease, color 0.15s ease"
        ]
        [ text q ]


-- ── Q&A pair ──────────────────────────────────────────────────────────────


viewQAPair : (Msg -> msg) -> Maybe String -> Maybe Int -> Int -> AnswerReport -> List (Html msg)
viewQAPair toMsg activeCitationId copiedIdx idx report =
    [ -- Question bubble
      div
        [ style "display" "flex"
        , style "justify-content" "flex-end"
        , class "anim-slide-right"
        ]
        [ div
            [ style "background" "#1e3a5f"
            , style "color" "#c7dff7"
            , style "border-radius" "14px 14px 3px 14px"
            , style "padding" "10px 16px"
            , style "font-size" "14px"
            , style "max-width" "72%"
            , style "line-height" "1.55"
            , style "font-weight" "500"
            ]
            [ text report.question ]
        ]

    -- Answer bubble
    , div
        [ style "display" "flex"
        , style "flex-direction" "column"
        , style "gap" "10px"
        , style "max-width" "88%"
        , class "anim-fade-up"
        ]
        [ -- Answer text
          div
            [ style "background" "#181818"
            , style "border" "1px solid #242424"
            , style "border-radius" "3px 14px 14px 14px"
            , style "padding" "14px 18px"
            , style "font-size" "14px"
            , style "line-height" "1.75"
            , style "color" "#c8c8c8"
            , style "position" "relative"
            ]
            [ Markdown.toHtml [ class "markdown-body" ] (cleanMarkdown report.answer)
            , -- Copy button (top-right corner)
              button
                [ onClick (toMsg (CopyAnswer idx report.answer))
                , style "position" "absolute"
                , style "top" "10px"
                , style "right" "10px"
                , style "background" (if copiedIdx == Just idx then "#14532d" else "#1e1e1e")
                , style "border" ("1px solid " ++ (if copiedIdx == Just idx then "#166534" else "#2e2e2e"))
                , style "border-radius" "6px"
                , style "padding" "3px 9px"
                , style "cursor" "pointer"
                , style "font-size" "11px"
                , style "font-weight" "600"
                , style "color" (if copiedIdx == Just idx then "#4ade80" else "#3a3a3a")
                , style "transition" "all 0.2s ease"
                ]
                [ text (if copiedIdx == Just idx then "✓ Copied" else "Copy") ]
            ]

        -- Confidence bar
        , confidenceBar (confidenceScore report)

        -- Citation pills — interactive
        , if List.isEmpty report.citations then
            text ""
          else
            div
                [ style "display" "flex"
                , style "flex-wrap" "wrap"
                , style "gap" "6px"
                ]
                (List.map (citationPill toMsg activeCitationId) report.citations)

        -- Timing
        , div
            [ style "font-size" "11px", style "color" "#2e2e2e" ]
            [ text (formatSeconds report.elapsedSeconds) ]
        ]
    ]


-- ── Confidence bar ────────────────────────────────────────────────────────


confidenceScore : AnswerReport -> Float
confidenceScore report =
    if List.isEmpty report.citations then
        0.5
    else
        List.map .confidence report.citations
            |> List.foldl (+) 0
            |> (\s -> s / toFloat (List.length report.citations))


confidenceBar : Float -> Html msg
confidenceBar score =
    let
        pct =
            round (score * 100)

        ( color, label ) =
            if score >= 0.8 then
                ( "#22c55e", "High confidence" )
            else if score >= 0.5 then
                ( "#f59e0b", "Medium confidence" )
            else
                ( "#f87171", "Low confidence" )
    in
    div
        [ style "display" "flex"
        , style "align-items" "center"
        , style "gap" "10px"
        ]
        [ div
            [ style "flex" "1"
            , style "height" "3px"
            , style "background" "#222"
            , style "border-radius" "2px"
            , style "overflow" "hidden"
            ]
            [ div
                [ style "width" (String.fromInt pct ++ "%")
                , style "height" "100%"
                , style "background" color
                , style "border-radius" "2px"
                , class "anim-grow-width"
                ]
                []
            ]
        , span
            [ style "font-size" "11px"
            , style "font-weight" "700"
            , style "color" color
            , style "white-space" "nowrap"
            , class "anim-fade-in"
            ]
            [ text (String.fromInt pct ++ "% · " ++ label) ]
        ]


-- ── Citation pill ─────────────────────────────────────────────────────────


citationPill : (Msg -> msg) -> Maybe String -> Citation -> Html msg
citationPill toMsg activeCitationId citation =
    let
        isActive =
            activeCitationId == Just citation.nodeId
    in
    button
        [ onClick (toMsg (CitationClicked citation.nodeId))
        , style "display" "inline-flex"
        , style "align-items" "center"
        , style "gap" "5px"
        , style "background" (if isActive then "#1e3a8a" else "#111827")
        , style "border" ("1px solid " ++ (if isActive then "#3b82f6" else "#1e2a3a"))
        , style "border-radius" "20px"
        , style "padding" "4px 11px"
        , style "cursor" "pointer"
        , style "transition" "all 0.15s ease"
        ]
        [ div
            [ style "width" "5px"
            , style "height" "5px"
            , style "border-radius" "50%"
            , style "background" (if isActive then "#60a5fa" else "#334155")
            , style "flex-shrink" "0"
            ]
            []
        , span
            [ style "font-size" "11px"
            , style "font-weight" "700"
            , style "color" (if isActive then "#93c5fd" else "#475569")
            , style "font-family" "monospace"
            ]
            [ text (leafNodeId citation.nodeId) ]
        ]


-- ── Thinking indicator ────────────────────────────────────────────────────


thinkingIndicator : String -> Html msg
thinkingIndicator docName =
    div
        [ style "display" "flex"
        , style "justify-content" "flex-start"
        , class "anim-fade-up"
        ]
        [ div
            [ style "background" "#181818"
            , style "border" "1px solid #242424"
            , style "border-radius" "3px 14px 14px 14px"
            , style "padding" "12px 18px"
            , style "display" "flex"
            , style "align-items" "center"
            , style "gap" "10px"
            ]
            [ div
                [ style "display" "flex"
                , style "gap" "4px"
                , style "align-items" "center"
                ]
                [ dot 0, dot 150, dot 300 ]
            , div []
                [ div
                    [ style "font-size" "13px"
                    , style "color" "#555"
                    , style "font-weight" "500"
                    ]
                    [ text "Analyzing contract" ]
                , div
                    [ style "font-size" "11px"
                    , style "color" "#333"
                    , style "margin-top" "2px"
                    ]
                    [ text docName ]
                ]
            ]
        ]


dot : Int -> Html msg
dot delayMs =
    div
        [ style "width" "5px"
        , style "height" "5px"
        , style "border-radius" "50%"
        , style "background" "#3a3a3a"
        , class "anim-pulse"
        , style "animation-delay" (String.fromInt delayMs ++ "ms")
        ]
        []


errorBubble : String -> Html msg
errorBubble err =
    div
        [ style "background" "#2d0a0a"
        , style "border" "1px solid #7f1d1d"
        , style "border-radius" "8px"
        , style "padding" "12px 16px"
        , style "font-size" "13px"
        , style "color" "#fca5a5"
        ]
        [ text err ]


-- ── Sources panel ─────────────────────────────────────────────────────────


viewSources : Model -> (Msg -> msg) -> Html msg
viewSources model toMsg =
    div
        [ style "width" "300px"
        , style "background" "#111"
        , style "border-left" "1px solid #1e1e1e"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "flex-shrink" "0"
        , style "height" "100%"
        ]
        [ -- Header
          div
            [ style "padding" "14px 16px 12px"
            , style "border-bottom" "1px solid #1a1a1a"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "space-between"
            ]
            [ div
                [ style "color" "#3a3a3a"
                , style "font-size" "10px"
                , style "font-weight" "700"
                , style "letter-spacing" "0.1em"
                , style "text-transform" "uppercase"
                ]
                [ text "Sources" ]
            , case model.report of
                Just report ->
                    span
                        [ style "font-size" "10px"
                        , style "color" "#2a2a2a"
                        , style "font-weight" "600"
                        ]
                        [ text (String.fromInt (List.length report.citations) ++ " cited") ]

                Nothing ->
                    text ""
            ]

        -- Source cards list
        , div
            [ style "flex" "1"
            , style "overflow-y" "auto"
            , style "padding" "10px 10px"
            , style "display" "flex"
            , style "flex-direction" "column"
            , style "gap" "8px"
            , class "anim-stagger"
            ]
            (case model.report of
                Nothing ->
                    [ div
                        [ style "color" "#222"
                        , style "font-size" "12px"
                        , style "line-height" "1.6"
                        , style "padding" "8px 4px"
                        , style "text-align" "center"
                        , style "margin-top" "24px"
                        ]
                        [ text "Sources will appear here after you ask a question." ]
                    ]

                Just report ->
                    if List.isEmpty report.citations then
                        [ div
                            [ style "color" "#2a2a2a"
                            , style "font-size" "12px"
                            , style "padding" "8px 4px"
                            ]
                            [ text "No citations found" ]
                        ]
                    else
                        List.map
                            (sourceCard toMsg model.activeCitationId model.expandedCitations)
                            report.citations
            )

        -- Hint when a citation is active
        , case model.activeCitationId of
            Just _ ->
                div
                    [ style "padding" "8px 12px"
                    , style "border-top" "1px solid #1a1a1a"
                    , style "font-size" "10px"
                    , style "color" "#2a2a2a"
                    , style "text-align" "center"
                    ]
                    [ text "Click citation again to deselect" ]

            Nothing ->
                text ""
        ]


sourceCard : (Msg -> msg) -> Maybe String -> Set String -> Citation -> Html msg
sourceCard toMsg activeCitationId expandedCitations citation =
    let
        conf =
            citation.confidence

        isActive =
            activeCitationId == Just citation.nodeId

        isExpanded =
            Set.member citation.nodeId expandedCitations

        theme =
            if conf >= 0.8 then
                { border = if isActive then "#22c55e" else "#14532d"
                , headerBg = if isActive then "#052e16" else "#0a1f0a"
                , label = if isActive then "#86efac" else "#4ade80"
                , bar = "#22c55e"
                }
            else if conf >= 0.5 then
                { border = if isActive then "#f59e0b" else "#78350f"
                , headerBg = if isActive then "#1c1000" else "#1c1200"
                , label = if isActive then "#fcd34d" else "#fbbf24"
                , bar = "#f59e0b"
                }
            else
                { border = if isActive then "#f87171" else "#7f1d1d"
                , headerBg = if isActive then "#200808" else "#1a0505"
                , label = if isActive then "#fca5a5" else "#f87171"
                , bar = "#ef4444"
                }

        pct =
            round (conf * 100)

        quoteText =
            citation.quote

        truncated =
            String.length quoteText > 160

        displayText =
            if isExpanded || not truncated then
                quoteText
            else
                String.left 157 quoteText ++ "…"
    in
    div
        [ id ("source-" ++ citation.nodeId)
        , style "background" (if isActive then "#161616" else "#141414")
        , style "border" ("1px solid " ++ theme.border)
        , style "border-radius" "8px"
        , style "overflow" "hidden"
        , style "transition" "border-color 0.2s ease, background 0.2s ease"
        ]
        [ -- Header
          div
            [ style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "space-between"
            , style "padding" "8px 11px"
            , style "background" theme.headerBg
            , style "border-bottom" ("1px solid " ++ theme.border)
            ]
            [ span
                [ style "font-size" "11px"
                , style "font-weight" "700"
                , style "color" theme.label
                , style "font-family" "monospace"
                ]
                [ text (leafNodeId citation.nodeId) ]
            , div
                [ style "display" "flex"
                , style "align-items" "center"
                , style "gap" "6px"
                ]
                [ div
                    [ style "width" "32px"
                    , style "height" "3px"
                    , style "background" "#1e1e1e"
                    , style "border-radius" "2px"
                    , style "overflow" "hidden"
                    ]
                    [ div
                        [ style "width" (String.fromInt pct ++ "%")
                        , style "height" "100%"
                        , style "background" theme.bar
                        ]
                        []
                    ]
                , span
                    [ style "font-size" "10px"
                    , style "font-weight" "700"
                    , style "color" theme.label
                    ]
                    [ text (String.fromInt pct ++ "%") ]
                ]
            ]

        -- Breadcrumb path
        , div
            [ style "padding" "7px 11px 0" ]
            [ div
                [ style "font-size" "10px"
                , style "color" "#2e2e2e"
                , style "font-weight" "600"
                , style "white-space" "nowrap"
                , style "overflow" "hidden"
                , style "text-overflow" "ellipsis"
                ]
                [ text (sourceBreadcrumb citation.nodePath) ]
            ]

        -- Quote body
        , div [ style "padding" "6px 11px 10px" ]
            [ div
                [ style "font-size" "12px"
                , style "color" "#555"
                , style "line-height" "1.6"
                , style "font-style" "italic"
                ]
                [ text ("\u{201C}" ++ displayText ++ "\u{201D}") ]
            , if truncated then
                button
                    [ onClick (toMsg (ToggleCitationExpand citation.nodeId))
                    , style "margin-top" "6px"
                    , style "background" "none"
                    , style "border" "none"
                    , style "padding" "0"
                    , style "cursor" "pointer"
                    , style "font-size" "11px"
                    , style "color" "#2a4a7a"
                    , style "font-weight" "600"
                    ]
                    [ text (if isExpanded then "Show less" else "Show more") ]
              else
                text ""
            ]
        ]


-- ── Helpers ───────────────────────────────────────────────────────────────


formatSeconds : Float -> String
formatSeconds s =
    let
        rounded =
            toFloat (round (s * 10)) / 10
    in
    String.fromFloat rounded ++ "s"


cleanMarkdown : String -> String
cleanMarkdown s =
    let
        trimmed =
            String.trimRight s

        stripTrailing str suffix =
            if String.endsWith suffix str then
                let
                    count =
                        List.length (String.split suffix str) - 1
                in
                if modBy 2 count == 1 then
                    String.trimRight (String.dropRight (String.length suffix) str)
                else
                    str
            else
                str
    in
    trimmed
        |> (\t -> stripTrailing t "**")
        |> (\t -> stripTrailing t "__")
        |> (\t -> stripTrailing t "*")
        |> (\t -> stripTrailing t "_")


leafNodeId : String -> String
leafNodeId nodeId =
    case List.reverse (String.split ">" nodeId) of
        last :: _ ->
            String.trim last

        [] ->
            nodeId


sourceBreadcrumb : List String -> String
sourceBreadcrumb path =
    path
        |> List.drop 1
        |> List.map String.trim
        |> String.join " › "
        |> (\s -> if String.isEmpty s then "Document" else s)


stripExtension : String -> String
stripExtension filename =
    let
        parts =
            String.split "." filename
    in
    case List.reverse parts of
        _ :: rest ->
            if List.isEmpty rest then
                filename
            else
                String.join "." (List.reverse rest)

        [] ->
            filename


enterToSubmit : msg -> JD.Decoder { message : msg, stopPropagation : Bool, preventDefault : Bool }
enterToSubmit msg =
    JD.field "keyCode" JD.int
        |> JD.andThen
            (\key ->
                if key == 13 then
                    JD.field "shiftKey" JD.bool
                        |> JD.andThen
                            (\shift ->
                                if shift then
                                    JD.fail "shift+enter"
                                else
                                    JD.succeed
                                        { message = msg
                                        , stopPropagation = False
                                        , preventDefault = True
                                        }
                            )
                else
                    JD.fail "not enter"
            )
