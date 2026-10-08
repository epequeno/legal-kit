module Main exposing (main)

import Browser
import Browser.Navigation as Nav
import Html exposing (Html)
import Json.Decode as Decode
import Page.Chat as Chat
import Page.Home as Home
import Url
import Url.Parser as Parser exposing ((</>), Parser)


main : Program () Model Msg
main =
    Browser.application
        { init = init
        , view = view
        , update = update
        , subscriptions = subscriptions
        , onUrlChange = UrlChanged
        , onUrlRequest = LinkClicked
        }



-- MODEL


type alias Model =
    { key : Nav.Key
    , route : Route
    , page : Page
    }


type Route
    = Home
    | ChatPage String
    | NotFound


type Page
    = HomePage Home.Model
    | ChatPageModel Chat.Model
    | NotFoundPage


init : () -> Url.Url -> Nav.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        route =
            parseUrl url

        ( page, cmd ) =
            initPage route
    in
    ( { key = key
      , route = route
      , page = page
      }
    , cmd
    )


initPage : Route -> ( Page, Cmd Msg )
initPage route =
    case route of
        Home ->
            let
                ( homeModel, homeCmd ) =
                    Home.init
            in
            ( HomePage homeModel, Cmd.map HomeMsg homeCmd )

        ChatPage docId ->
            let
                ( chatModel, chatCmd ) =
                    Chat.init docId
            in
            ( ChatPageModel chatModel, Cmd.map ChatMsg chatCmd )

        NotFound ->
            ( NotFoundPage, Cmd.none )



-- ROUTING


routeParser : Parser (Route -> a) a
routeParser =
    Parser.oneOf
        [ Parser.map Home Parser.top
        , Parser.map ChatPage (Parser.s "documents" </> Parser.string)
        ]


parseUrl : Url.Url -> Route
parseUrl url =
    Parser.parse routeParser url
        |> Maybe.withDefault NotFound



-- UPDATE


type Msg
    = LinkClicked Browser.UrlRequest
    | UrlChanged Url.Url
    | HomeMsg Home.Msg
    | ChatMsg Chat.Msg
    | NavigateToChat String
    | NavigateToHome


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        LinkClicked urlRequest ->
            case urlRequest of
                Browser.Internal url ->
                    ( model, Nav.pushUrl model.key (Url.toString url) )

                Browser.External href ->
                    ( model, Nav.load href )

        UrlChanged url ->
            let
                route =
                    parseUrl url

                ( page, cmd ) =
                    initPage route
            in
            ( { model | route = route, page = page }, cmd )

        HomeMsg homeMsg ->
            case model.page of
                HomePage homeModel ->
                    let
                        ( newHomeModel, homeCmd ) =
                            Home.update homeMsg homeModel

                        -- Check if we need to navigate
                        navCmd =
                            case homeMsg of
                                Home.NavigateToChat docId ->
                                    Nav.pushUrl model.key ("/documents/" ++ docId)

                                _ ->
                                    Cmd.none
                    in
                    ( { model | page = HomePage newHomeModel }
                    , Cmd.batch [ Cmd.map HomeMsg homeCmd, navCmd ]
                    )

                _ ->
                    ( model, Cmd.none )

        ChatMsg chatMsg ->
            case model.page of
                ChatPageModel chatModel ->
                    let
                        ( newChatModel, chatCmd ) =
                            Chat.update chatMsg chatModel

                        -- Check if we need to navigate
                        navCmd =
                            case chatMsg of
                                Chat.NavigateBack ->
                                    Nav.pushUrl model.key "/"

                                _ ->
                                    Cmd.none
                    in
                    ( { model | page = ChatPageModel newChatModel }
                    , Cmd.batch [ Cmd.map ChatMsg chatCmd, navCmd ]
                    )

                _ ->
                    ( model, Cmd.none )

        NavigateToChat docId ->
            ( model, Nav.pushUrl model.key ("/documents/" ++ docId) )

        NavigateToHome ->
            ( model, Nav.pushUrl model.key "/" )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    case model.page of
        HomePage _ ->
            Sub.none

        ChatPageModel _ ->
            Sub.map ChatMsg (Chat.askEvent Chat.AskEvent)

        NotFoundPage ->
            Sub.none



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "LegalKit"
    , body =
        [ case model.page of
            HomePage homeModel ->
                Home.view homeModel HomeMsg NavigateToChat

            ChatPageModel chatModel ->
                Chat.view chatModel ChatMsg (\_ -> NavigateToHome)

            NotFoundPage ->
                Html.div [] [ Html.text "404 - Page not found" ]
        ]
    }
