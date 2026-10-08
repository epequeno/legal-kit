module Components.DocumentCard exposing (view)

import Html exposing (Html, button, div, span, text)
import Html.Attributes exposing (class, style, title)
import Html.Events exposing (onClick, stopPropagationOn)
import Json.Decode as Decode
import Types exposing (Document)


view : Document -> (String -> msg) -> (String -> msg) -> Bool -> Html msg
view doc onChat onDelete isActive =
    div
        [ onClick (onChat doc.id)
        , style "border-radius" "8px"
        , style "padding" "10px 11px"
        , style "cursor" "pointer"
        , style "background" (if isActive then "#1e1e1e" else "transparent")
        , style "border" ("1px solid " ++ (if isActive then "#2a2a2a" else "transparent"))
        , style "transition" "background 0.12s ease, border-color 0.12s ease"
        , style "display" "flex"
        , style "align-items" "center"
        , style "gap" "10px"
        , class "hover-bg"
        ]
        [ -- Doc icon
          div
            [ style "width" "28px"
            , style "height" "34px"
            , style "background" (if isActive then "#1e3a5f" else "#1a1a1a")
            , style "border" ("1px solid " ++ (if isActive then "#2563eb" else "#252525"))
            , style "border-radius" "4px"
            , style "flex-shrink" "0"
            , style "display" "flex"
            , style "align-items" "center"
            , style "justify-content" "center"
            , style "transition" "all 0.12s ease"
            ]
            [ span
                [ style "font-size" "10px"
                , style "font-weight" "800"
                , style "color" (if isActive then "#60a5fa" else "#333")
                ]
                [ text (fileTypeLabel doc.filename) ]
            ]

        -- Name + meta
        , div [ style "flex" "1", style "min-width" "0" ]
            [ div
                [ style "font-size" "12px"
                , style "font-weight" "600"
                , style "color" (if isActive then "#e0e0e0" else "#666")
                , style "white-space" "nowrap"
                , style "overflow" "hidden"
                , style "text-overflow" "ellipsis"
                , style "line-height" "1.3"
                , style "transition" "color 0.12s ease"
                ]
                [ text (stripExtension doc.filename) ]
            , div
                [ style "font-size" "10px"
                , style "color" (if isActive then "#3a3a3a" else "#282828")
                , style "margin-top" "3px"
                ]
                [ text (String.fromInt doc.nChunks ++ " sections") ]
            ]

        -- Delete button
        , button
            [ stopPropagationOn "click" (Decode.succeed ( onDelete doc.id, True ))
            , title "Remove"
            , style "background" "none"
            , style "border" "none"
            , style "cursor" "pointer"
            , style "color" "#222"
            , style "font-size" "15px"
            , style "padding" "2px 3px"
            , style "flex-shrink" "0"
            , style "line-height" "1"
            , style "opacity" "0"
            , style "transition" "opacity 0.12s ease, color 0.12s ease"
            , class "doc-delete-btn"
            ]
            [ text "×" ]
        ]


fileTypeLabel : String -> String
fileTypeLabel filename =
    case String.toLower (fileExtension filename) of
        "pdf" -> "PDF"
        "docx" -> "DOC"
        "txt" -> "TXT"
        _ -> "DOC"


fileExtension : String -> String
fileExtension filename =
    case List.reverse (String.split "." filename) of
        ext :: _ -> ext
        [] -> ""


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
