module Main exposing (main)

import Browser
import Html exposing (Html, button, div, span, text)
import Html.Attributes exposing (class, href, style, target)
import Html.Events exposing (onClick)
import Json.Decode as Decode
import Page.Chat as Chat
import Page.Home as Home
import Page.Settings as Settings
import Ports
import Set
import Svg exposing (circle, path, rect, svg)
import Svg.Attributes as SA
import Types exposing (ApiKeyStatus, Document)


-- ── Model ─────────────────────────────────────────────────────────────────


type Page
    = MainView
    | SettingsView


type alias Model =
    { home : Home.Model
    , activeDoc : Maybe Document
    , chat : Maybe Chat.Model
    , settings : Maybe Settings.Model
    , page : Page
    }


-- ── Init ──────────────────────────────────────────────────────────────────


init : () -> ( Model, Cmd Msg )
init _ =
    let
        ( homeModel, homeCmd ) =
            Home.init
    in
    ( { home = homeModel
      , activeDoc = Nothing
      , chat = Nothing
      , settings = Nothing
      , page = MainView
      }
    , Cmd.map HomeMsg homeCmd
    )


-- ── Msg ───────────────────────────────────────────────────────────────────


type Msg
    = HomeMsg Home.Msg
    | ChatMsg Chat.Msg
    | SettingsMsg Settings.Msg
    | OpenSettings
    | CloseSettings
    | TauriIngestResult Decode.Value
    | TauriAskResult Decode.Value
    | TauriListResult Decode.Value
    | TauriDeleteResult Decode.Value
    | TauriApiKeyStatus Decode.Value
    | TauriChatHistory Decode.Value
    | TauriDocumentOutline Decode.Value
    | TauriSettingsResult Decode.Value
    | TauriSaveSettingsDone ()
    | TauriClearAllDone ()
    | TauriFileSelected String
    | TauriError String


-- ── Update ────────────────────────────────────────────────────────────────


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        OpenSettings ->
            let
                ( settingsModel, settingsCmd ) =
                    Settings.init
            in
            ( { model | page = SettingsView, settings = Just settingsModel }
            , Cmd.map SettingsMsg settingsCmd
            )

        CloseSettings ->
            -- Refresh API key status when leaving settings
            ( { model | page = MainView, settings = Nothing }
            , Cmd.batch
                [ Ports.invokeGetApiKeyStatus ()
                , Ports.invokeListDocuments ()
                ]
            )

        HomeMsg subMsg ->
            case subMsg of
                Home.NavigateToChat docId ->
                    let
                        docOpt =
                            model.home.documents
                                |> List.filter (\d -> d.id == docId)
                                |> List.head
                    in
                    case docOpt of
                        Just doc ->
                            let
                                ( chatModel, chatCmd ) =
                                    Chat.init doc
                            in
                            ( { model | activeDoc = Just doc, chat = Just chatModel, page = MainView }
                            , Cmd.map ChatMsg chatCmd
                            )

                        Nothing ->
                            ( model, Cmd.none )

                _ ->
                    let
                        ( newHome, homeCmd ) =
                            Home.update subMsg model.home
                    in
                    ( { model | home = newHome }, Cmd.map HomeMsg homeCmd )

        ChatMsg subMsg ->
            case model.chat of
                Nothing ->
                    ( model, Cmd.none )

                Just chatModel ->
                    case subMsg of
                        Chat.NavigateHome ->
                            ( { model | activeDoc = Nothing, chat = Nothing }, Cmd.none )

                        _ ->
                            let
                                ( newChat, chatCmd ) =
                                    Chat.update subMsg chatModel
                            in
                            ( { model | chat = Just newChat }, Cmd.map ChatMsg chatCmd )

        SettingsMsg subMsg ->
            case model.settings of
                Nothing ->
                    ( model, Cmd.none )

                Just settingsModel ->
                    case subMsg of
                        Settings.NavigateHome ->
                            update CloseSettings model

                        _ ->
                            let
                                ( newSettings, settingsCmd ) =
                                    Settings.update subMsg settingsModel
                            in
                            ( { model | settings = Just newSettings }, Cmd.map SettingsMsg settingsCmd )

        -- Tauri inbound messages
        TauriFileSelected path ->
            let
                ( newHome, homeCmd ) =
                    Home.update (Home.FileSelected path) model.home
            in
            ( { model | home = newHome }, Cmd.map HomeMsg homeCmd )

        TauriIngestResult val ->
            let
                ( newHome, homeCmd ) =
                    Home.update (Home.GotIngestResult val) model.home
            in
            ( { model | home = newHome }, Cmd.map HomeMsg homeCmd )

        TauriListResult val ->
            let
                ( newHome, homeCmd ) =
                    Home.update (Home.GotListResult val) model.home
            in
            ( { model | home = newHome }, Cmd.map HomeMsg homeCmd )

        TauriDeleteResult val ->
            let
                ( newHome, homeCmd ) =
                    Home.update (Home.GotDeleteResult val) model.home
            in
            ( { model | home = newHome }, Cmd.map HomeMsg homeCmd )

        TauriApiKeyStatus val ->
            let
                ( newHome, homeCmd ) =
                    Home.update (Home.GotApiKeyStatus val) model.home

                settingsUpdate =
                    case model.settings of
                        Just settingsModel ->
                            let
                                ( newSettings, settingsCmd ) =
                                    Settings.update (Settings.GotApiStatus val) settingsModel
                            in
                            ( Just newSettings, Cmd.map SettingsMsg settingsCmd )

                        Nothing ->
                            ( model.settings, Cmd.none )
            in
            ( { model | home = newHome, settings = Tuple.first settingsUpdate }
            , Cmd.batch [ Cmd.map HomeMsg homeCmd, Tuple.second settingsUpdate ]
            )

        TauriAskResult val ->
            case model.chat of
                Nothing ->
                    ( model, Cmd.none )

                Just chatModel ->
                    let
                        ( newChat, chatCmd ) =
                            Chat.update (Chat.GotAnswer val) chatModel
                    in
                    ( { model | chat = Just newChat }, Cmd.map ChatMsg chatCmd )

        TauriChatHistory val ->
            case model.chat of
                Nothing ->
                    ( model, Cmd.none )

                Just chatModel ->
                    let
                        ( newChat, chatCmd ) =
                            Chat.update (Chat.GotHistoryResult val) chatModel
                    in
                    ( { model | chat = Just newChat }, Cmd.map ChatMsg chatCmd )

        TauriDocumentOutline val ->
            case model.chat of
                Nothing ->
                    ( model, Cmd.none )

                Just chatModel ->
                    let
                        ( newChat, chatCmd ) =
                            Chat.update (Chat.GotOutline val) chatModel
                    in
                    ( { model | chat = Just newChat }, Cmd.map ChatMsg chatCmd )

        TauriSettingsResult val ->
            case model.settings of
                Nothing ->
                    ( model, Cmd.none )

                Just settingsModel ->
                    let
                        ( newSettings, settingsCmd ) =
                            Settings.update (Settings.GotSettings val) settingsModel
                    in
                    ( { model | settings = Just newSettings }, Cmd.map SettingsMsg settingsCmd )

        TauriSaveSettingsDone _ ->
            case model.settings of
                Nothing ->
                    ( model, Cmd.none )

                Just settingsModel ->
                    let
                        ( newSettings, settingsCmd ) =
                            Settings.update Settings.SaveDone settingsModel
                    in
                    ( { model | settings = Just newSettings }
                    , Cmd.batch [ Cmd.map SettingsMsg settingsCmd, Ports.invokeGetApiKeyStatus () ]
                    )

        TauriClearAllDone _ ->
            case model.settings of
                Nothing ->
                    ( model, Cmd.none )

                Just settingsModel ->
                    let
                        ( newSettings, settingsCmd ) =
                            Settings.update Settings.ClearAllDone settingsModel

                        ( newHome, homeCmd ) =
                            Home.init
                    in
                    ( { model
                        | settings = Just newSettings
                        , home = newHome
                        , chat = Nothing
                        , activeDoc = Nothing
                      }
                    , Cmd.batch [ Cmd.map SettingsMsg settingsCmd, Cmd.map HomeMsg homeCmd ]
                    )

        TauriError err ->
            case model.page of
                SettingsView ->
                    case model.settings of
                        Just settingsModel ->
                            let
                                ( newSettings, _ ) =
                                    Settings.update (Settings.GotError err) settingsModel
                            in
                            ( { model | settings = Just newSettings }, Cmd.none )

                        Nothing ->
                            ( model, Cmd.none )

                MainView ->
                    let
                        ( newHome, homeCmd ) =
                            Home.update (Home.GotError err) model.home
                    in
                    ( { model | home = newHome }, Cmd.map HomeMsg homeCmd )


-- ── View ──────────────────────────────────────────────────────────────────


view : Model -> Html Msg
view model =
    div
        [ style "width" "100vw"
        , style "height" "100vh"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "background" "#141414"
        , style "font-family" "-apple-system, BlinkMacSystemFont, 'Inter', sans-serif"
        , style "color" "#d4d4d4"
        , style "overflow" "hidden"
        ]
        [ div
            [ style "flex" "1"
            , style "display" "flex"
            , style "overflow" "hidden"
            ]
            [ -- Sidebar (always visible)
              sidebarView model

            -- Main content area
            , case model.page of
                SettingsView ->
                    case model.settings of
                        Just settingsModel ->
                            Settings.view settingsModel SettingsMsg

                        Nothing ->
                            text ""

                MainView ->
                    div
                        [ style "flex" "1"
                        , style "display" "flex"
                        , style "overflow" "hidden"
                        ]
                        [ -- Q&A panel
                          div
                            [ style "flex" "1"
                            , style "display" "flex"
                            , style "flex-direction" "column"
                            , style "background" "#141414"
                            , style "min-width" "0"
                            ]
                            [ case model.chat of
                                Just chatModel ->
                                    Chat.view chatModel ChatMsg (ChatMsg Chat.NavigateHome)

                                Nothing ->
                                    emptyState
                                        { loading = model.home.loading
                                        , noDocs = List.isEmpty model.home.documents
                                        , apiKeys = model.home.apiKeys
                                        }
                                        (HomeMsg Home.OpenFilePicker)
                                        OpenSettings
                            ]

                        -- Sources panel
                        , Chat.viewSources (Maybe.withDefault emptyChat model.chat) ChatMsg
                        ]
            ]
        ]


sidebarView : Model -> Html Msg
sidebarView model =
    -- Home.view already renders the full 220px sidebar.
    -- We just pass it through. The settings button is added inside Home.view
    -- via the onSettings callback.
    Home.view model.home HomeMsg (\docId -> HomeMsg (Home.NavigateToChat docId)) OpenSettings (model.page == SettingsView)


emptyChat : Chat.Model
emptyChat =
    { doc = { id = "", filename = "", originalPath = "", sizeBytes = 0, nChunks = 0, ingestedAt = "" }
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


type alias EmptyStateCtx =
    { loading : Bool
    , noDocs : Bool
    , apiKeys : Maybe ApiKeyStatus
    }


emptyState : EmptyStateCtx -> Msg -> Msg -> Html Msg
emptyState ctx onOpenFile onOpenSettings =
    let
        apiReady =
            ctx.apiKeys |> Maybe.map .ready |> Maybe.withDefault False
    in
    if ctx.loading then
        loadingState

    else if not apiReady && ctx.noDocs then
        setupGuide onOpenFile onOpenSettings ctx.apiKeys

    else if ctx.noDocs then
        heroState onOpenFile

    else
        selectState


-- ── Loading ───────────────────────────────────────────────────────────────


loadingState : Html Msg
loadingState =
    div
        [ style "flex" "1"
        , style "display" "flex"
        , style "align-items" "center"
        , style "justify-content" "center"
        ]
        [ div [ style "font-size" "13px", style "color" "#2a2a2a" ] [ text "Loading…" ] ]


-- ── Setup guide (no API key) ──────────────────────────────────────────────


setupGuide : Msg -> Msg -> Maybe ApiKeyStatus -> Html Msg
setupGuide onOpenFile onOpenSettings apiKeys =
    let
        step1Done =
            apiKeys |> Maybe.map .ready |> Maybe.withDefault False
    in
    div
        [ style "flex" "1"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "align-items" "center"
        , style "justify-content" "center"
        , style "gap" "40px"
        , style "padding" "48px"
        , class "anim-fade-in"
        ]
        [ -- App mark + name
          div [ style "text-align" "center", style "display" "flex", style "flex-direction" "column", style "align-items" "center", style "gap" "14px" ]
            [ appIcon 40
            , div []
                [ div [ style "font-size" "22px", style "font-weight" "800", style "color" "#e0e0e0", style "letter-spacing" "-0.5px" ]
                    [ text "LegalKit" ]
                , div [ style "font-size" "13px", style "color" "#444", style "margin-top" "4px" ]
                    [ text "Ask questions about contracts. Get cited answers." ]
                ]
            ]

        -- Steps
        , div [ style "display" "flex", style "flex-direction" "column", style "gap" "12px", style "width" "100%", style "max-width" "420px" ]
            [ setupStep "1" "Get an API key" "Free tier available — no credit card required." True
                (Html.a
                    [ href "https://openrouter.ai/keys"
                    , target "_blank"
                    , style "display" "inline-block"
                    , style "margin-top" "10px"
                    , style "background" "#2563eb"
                    , style "color" "#fff"
                    , style "font-size" "12px"
                    , style "font-weight" "700"
                    , style "padding" "6px 14px"
                    , style "border-radius" "6px"
                    , style "text-decoration" "none"
                    ]
                    [ text "openrouter.ai →" ]
                )
            , setupStep "2" "Add it in Settings" "Paste your OpenRouter API key to unlock AI answers." (not step1Done)
                (button
                    [ onClick onOpenSettings
                    , style "margin-top" "10px"
                    , style "background" "#1e1e1e"
                    , style "border" "1px solid #2e2e2e"
                    , style "color" "#aaa"
                    , style "font-size" "12px"
                    , style "font-weight" "700"
                    , style "padding" "6px 14px"
                    , style "border-radius" "6px"
                    , style "cursor" "pointer"
                    ]
                    [ text "Open Settings" ]
                )
            , setupStep "3" "Open a contract" "Supports PDF, DOCX, and plain text." True
                (button
                    [ onClick onOpenFile
                    , style "margin-top" "10px"
                    , style "background" "#1e1e1e"
                    , style "border" "1px solid #2e2e2e"
                    , style "color" "#aaa"
                    , style "font-size" "12px"
                    , style "font-weight" "700"
                    , style "padding" "6px 14px"
                    , style "border-radius" "6px"
                    , style "cursor" "pointer"
                    ]
                    [ text "Choose File" ]
                )
            ]
        ]


setupStep : String -> String -> String -> Bool -> Html Msg -> Html Msg
setupStep num title_ desc showAction action =
    div
        [ style "display" "flex"
        , style "gap" "16px"
        , style "align-items" "flex-start"
        ]
        [ div
            [ style "width" "26px"
            , style "height" "26px"
            , style "border-radius" "50%"
            , style "background" "#1e1e1e"
            , style "border" "1px solid #2e2e2e"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "center"
            , style "flex-shrink" "0"
            , style "margin-top" "2px"
            ]
            [ span [ style "font-size" "11px", style "font-weight" "800", style "color" "#555" ] [ text num ] ]
        , div [ style "flex" "1" ]
            [ div [ style "font-size" "14px", style "font-weight" "700", style "color" "#c0c0c0" ] [ text title_ ]
            , div [ style "font-size" "12px", style "color" "#444", style "margin-top" "3px", style "line-height" "1.5" ] [ text desc ]
            , if showAction then action else text ""
            ]
        ]


-- ── Hero (API ready, no docs) ─────────────────────────────────────────────


heroState : Msg -> Html Msg
heroState onOpenFile =
    div
        [ style "flex" "1"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "align-items" "center"
        , style "justify-content" "center"
        , style "gap" "36px"
        , style "padding" "48px"
        , class "anim-fade-in"
        ]
        [ -- Logo + name
          div [ style "text-align" "center", style "display" "flex", style "flex-direction" "column", style "align-items" "center", style "gap" "16px" ]
            [ appIcon 44
            , div []
                [ div [ style "font-size" "26px", style "font-weight" "800", style "color" "#e8e8e8", style "letter-spacing" "-0.5px" ]
                    [ text "LegalKit" ]
                , div [ style "font-size" "14px", style "color" "#444", style "margin-top" "6px", style "line-height" "1.6" ]
                    [ text "Ask questions about contracts." ]
                , div [ style "font-size" "14px", style "color" "#444" ]
                    [ text "Get precise, cited answers." ]
                ]
            ]

        -- Capability pills
        , div [ style "display" "flex", style "gap" "8px", style "flex-wrap" "wrap", style "justify-content" "center" ]
            [ capPill "Hybrid search"
            , capPill "Cited sources"
            , capPill "Confidence scoring"
            , capPill "Local AI"
            ]

        -- CTA
        , div [ style "display" "flex", style "flex-direction" "column", style "align-items" "center", style "gap" "10px" ]
            [ button
                [ onClick onOpenFile
                , style "background" "#2563eb"
                , style "border" "none"
                , style "border-radius" "10px"
                , style "padding" "13px 32px"
                , style "cursor" "pointer"
                , style "font-size" "14px"
                , style "font-weight" "700"
                , style "color" "#fff"
                , style "letter-spacing" "0.01em"
                ]
                [ text "Open a Contract" ]
            , div [ style "font-size" "11px", style "color" "#2e2e2e" ]
                [ text "PDF · DOCX · TXT" ]
            ]
        ]


capPill : String -> Html Msg
capPill label =
    div
        [ style "background" "#181818"
        , style "border" "1px solid #242424"
        , style "border-radius" "20px"
        , style "padding" "5px 13px"
        , style "font-size" "11px"
        , style "font-weight" "600"
        , style "color" "#3a3a3a"
        ]
        [ text label ]


-- ── Select prompt (has docs, none active) ────────────────────────────────


selectState : Html Msg
selectState =
    div
        [ style "flex" "1"
        , style "display" "flex"
        , style "flex-direction" "column"
        , style "align-items" "center"
        , style "justify-content" "center"
        , style "gap" "12px"
        , class "anim-fade-in"
        ]
        [ div
            [ style "display" "flex"
            , style "align-items" "center"
            , style "gap" "10px"
            , style "color" "#2e2e2e"
            ]
            [ span [ style "font-size" "18px" ] [ text "←" ]
            , span [ style "font-size" "13px", style "font-weight" "500" ] [ text "Select a contract to begin" ]
            ]
        ]


-- ── App icon SVG ──────────────────────────────────────────────────────────


appIcon : Int -> Html Msg
appIcon size =
    let
        s =
            String.fromInt size
    in
    svg
        [ SA.width s
        , SA.height s
        , SA.viewBox "0 0 48 48"
        , SA.fill "none"
        ]
        [ -- Document body
          rect
            [ SA.x "6", SA.y "4", SA.width "28", SA.height "36"
            , SA.rx "4"
            , SA.fill "#1e3a5f"
            , SA.stroke "#2563eb"
            , SA.strokeWidth "1.5"
            ]
            []
        -- Fold corner
        , path
            [ SA.d "M28 4 L34 10 L28 10 Z"
            , SA.fill "#2563eb"
            , SA.opacity "0.6"
            ]
            []
        -- Text lines
        , rect [ SA.x "12", SA.y "18", SA.width "16", SA.height "1.5", SA.rx "0.75", SA.fill "#3b82f6", SA.opacity "0.5" ] []
        , rect [ SA.x "12", SA.y "23", SA.width "16", SA.height "1.5", SA.rx "0.75", SA.fill "#3b82f6", SA.opacity "0.5" ] []
        , rect [ SA.x "12", SA.y "28", SA.width "10", SA.height "1.5", SA.rx "0.75", SA.fill "#3b82f6", SA.opacity "0.5" ] []
        -- Sparkle badge
        , circle [ SA.cx "37", SA.cy "37", SA.r "9", SA.fill "#111827", SA.stroke "#1d4ed8", SA.strokeWidth "1.5" ] []
        , path
            [ SA.d "M37 31 L38.2 35.2 L42 36 L38.2 36.8 L37 41 L35.8 36.8 L32 36 L35.8 35.2 Z"
            , SA.fill "#60a5fa"
            ]
            []
        ]


text : String -> Html msg
text =
    Html.text


-- ── Subscriptions ─────────────────────────────────────────────────────────


subscriptions : Model -> Sub Msg
subscriptions _ =
    Sub.batch
        [ Ports.onFileSelected TauriFileSelected
        , Ports.onIngestResult TauriIngestResult
        , Ports.onAskResult TauriAskResult
        , Ports.onListResult TauriListResult
        , Ports.onDeleteResult TauriDeleteResult
        , Ports.onApiKeyStatus TauriApiKeyStatus
        , Ports.onChatHistoryResult TauriChatHistory
        , Ports.onDocumentOutline TauriDocumentOutline
        , Ports.onSettingsResult TauriSettingsResult
        , Ports.onSaveSettingsDone TauriSaveSettingsDone
        , Ports.onClearAllDone TauriClearAllDone
        , Ports.onTauriError TauriError
        ]


-- ── Main ──────────────────────────────────────────────────────────────────


main : Program () Model Msg
main =
    Browser.element
        { init = init
        , update = update
        , view = view
        , subscriptions = subscriptions
        }
