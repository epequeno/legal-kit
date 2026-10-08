module Components.CitationCard exposing (view)

import Html exposing (Html, button, div, span, text)
import Html.Attributes exposing (class, style)
import Html.Events exposing (onClick)
import Types exposing (Citation, Claim)


view : Claim -> Bool -> msg -> Html msg
view claim expanded toggleMsg =
    div
        [ style "background" "#f9fafb"
        , style "border" "1px solid #e5e7eb"
        , style "border-radius" "6px"
        , style "padding" "16px"
        , style "margin-bottom" "12px"
        ]
        [ div [ style "display" "flex", style "justify-content" "space-between", style "align-items" "flex-start" ]
            [ div [ style "flex" "1" ]
                [ div
                    [ style "font-size" "14px"
                    , style "color" "#374151"
                    , style "margin-bottom" "8px"
                    , style "line-height" "1.5"
                    ]
                    [ text claim.text ]
                , div [ style "display" "flex", style "gap" "8px", style "flex-wrap" "wrap" ]
                    [ statusBadge claim
                    , confidenceBadge claim.confidence
                    , if claim.isGap then
                        gapBadge

                      else
                        text ""
                    , if claim.hasHedging then
                        hedgingBadge

                      else
                        text ""
                    ]
                ]
            , button
                [ onClick toggleMsg
                , style "background" "transparent"
                , style "border" "none"
                , style "color" "#6b7280"
                , style "cursor" "pointer"
                , style "font-size" "12px"
                , style "padding" "4px 8px"
                ]
                [ text
                    (if expanded then
                        "▼ Hide citations"

                     else
                        "▶ Show citations (" ++ String.fromInt (List.length claim.citations) ++ ")"
                    )
                ]
            ]
        , if expanded then
            citationsList claim.citations

          else
            text ""
        ]


statusBadge : Claim -> Html msg
statusBadge claim =
    let
        ( color, label ) =
            if List.isEmpty claim.citations then
                ( "#9ca3af", "No citations" )

            else
                let
                    verifiedCount =
                        List.length (List.filter (\c -> c.status == "verified") claim.citations)

                    totalCount =
                        List.length claim.citations
                in
                if verifiedCount == totalCount then
                    ( "#16a34a", "Verified" )

                else if verifiedCount > 0 then
                    ( "#d97706", "Partial" )

                else
                    ( "#dc2626", "Unverified" )
    in
    span
        [ style "background" color
        , style "color" "white"
        , style "padding" "2px 8px"
        , style "border-radius" "4px"
        , style "font-size" "12px"
        , style "font-weight" "500"
        ]
        [ text label ]


confidenceBadge : Types.Confidence -> Html msg
confidenceBadge conf =
    span
        [ style "background" "#e5e7eb"
        , style "color" "#374151"
        , style "padding" "2px 8px"
        , style "border-radius" "4px"
        , style "font-size" "12px"
        , style "font-weight" "500"
        ]
        [ text (conf.level ++ " (" ++ String.fromInt (round (conf.score * 100)) ++ "%)") ]


gapBadge : Html msg
gapBadge =
    span
        [ style "background" "#fbbf24"
        , style "color" "#78350f"
        , style "padding" "2px 8px"
        , style "border-radius" "4px"
        , style "font-size" "12px"
        , style "font-weight" "500"
        ]
        [ text "Gap" ]


hedgingBadge : Html msg
hedgingBadge =
    span
        [ style "background" "#cbd5e1"
        , style "color" "#334155"
        , style "padding" "2px 8px"
        , style "border-radius" "4px"
        , style "font-size" "12px"
        , style "font-weight" "500"
        ]
        [ text "Hedged" ]


citationsList : List Citation -> Html msg
citationsList citations =
    div
        [ style "margin-top" "12px"
        , style "padding-top" "12px"
        , style "border-top" "1px solid #e5e7eb"
        ]
        (if List.isEmpty citations then
            [ div [ style "font-size" "13px", style "color" "#6b7280", style "font-style" "italic" ]
                [ text "No citations available" ]
            ]

         else
            List.map citationItem citations
        )


citationItem : Citation -> Html msg
citationItem citation =
    div
        [ style "background" "white"
        , style "border" "1px solid #e5e7eb"
        , style "border-radius" "4px"
        , style "padding" "12px"
        , style "margin-bottom" "8px"
        ]
        [ div
            [ style "font-size" "12px"
            , style "color" "#6b7280"
            , style "font-weight" "600"
            , style "margin-bottom" "6px"
            ]
            [ text ("Section: " ++ citation.sectionId) ]
        , div
            [ style "font-size" "13px"
            , style "color" "#374151"
            , style "line-height" "1.5"
            , style "margin-bottom" "8px"
            , style "font-style" "italic"
            ]
            [ text ("\"" ++ citation.sourceText ++ "\"") ]
        , div [ style "display" "flex", style "gap" "8px", style "font-size" "12px" ]
            [ span
                [ style "color" "#6b7280" ]
                [ text ("Entailment: " ++ citation.entailment) ]
            , span
                [ style "color" "#9ca3af" ]
                [ text ("Score: " ++ String.fromFloat citation.entailmentScore) ]
            ]
        ]
