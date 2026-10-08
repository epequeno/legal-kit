# ADR-011: UI Design Direction

## Status

Accepted

## Date

2026-03-19

## Context

With the pivot to a native Mac app, the web app's single-panel layout (white card, upload widget)
was carried over unchanged. A visual design session explored three layout directions:

- **Option A — Sidebar**: Dark sidebar + light chat panel (Mail-like)
- **Option B — Focused**: Centered single-column, full-width, minimal (iA Writer-like)
- **Option C — Three Panel**: Doc list + Q&A + Sources side-by-side

The primary use case is a lawyer asking a question about a specific contract and needing to
immediately assess how confident the answer is and which clauses it's based on.

## Decision

**Three-panel layout (Option C) with dark mode as the default theme.**

### Layout

```
┌─────────────┬──────────────────────────┬────────────────┐
│  Contracts  │       Q & A              │    Sources     │
│  220px      │       flex:1             │    300px       │
│             │                          │                │
│  doc list   │  question bubbles        │  cited clause  │
│  sidebar    │  answer + confidence     │  cards with    │
│             │  ambiguity cards         │  confidence %  │
│  + Open     │  [   ask input   ] [Ask] │                │
└─────────────┴──────────────────────────┴────────────────┘
```

### Confidence surfacing

Two states:

**High confidence (≥70%)**: Single answer with a confidence bar and citation pills inline.

**Ambiguous (<70% or conflicting sources)**: Amber warning banner + two "Interpretation" cards,
each with its own confidence percentage and supporting citation. Teaches the user the contract
is genuinely ambiguous rather than hiding it.

### Source panel states

Each source card has three visual states based on its relevance confidence:

| State | Color | Meaning |
|---|---|---|
| High (≥80%) | Green (`#22c55e`) | Strongly supports the answer |
| Medium (40-79%) | Amber (`#f59e0b`) | Partial match or indirect support |
| Conflicting | Red (`#f87171`) | Contradicts another source — user must be aware |

### Theme

**Dark by default**, light mode available as a toggle (persisted in user preferences).

Dark palette:
```
Background:   #141414
Surface:      #1e1e1e
Sidebar:      #1a1a1a
Sources:      #111111
Border:       #2a2a2a
Text primary: #f0f0f0
Text muted:   #777777
Accent:       #2563eb
```

Light palette (same structure, inverted):
```
Background:   #f5f4ee
Surface:      #ffffff
Border:       #e5e5e0
Text primary: #1a1a1a
```

Accent, green, amber, and red are the same in both modes.

### Typography

System font stack: `-apple-system, BlinkMacSystemFont, "Inter", sans-serif`
Monospace (section IDs): `"SF Mono", "JetBrains Mono", monospace`

### Implementation in Elm

Theme is a `type` passed through the view hierarchy:

```elm
type Theme = Dark | Light

view : Theme -> Model -> Html Msg
```

Color values are expressed as CSS custom properties set on `:root` and toggled by a `data-theme`
attribute on `<body>`. No style recompilation needed to switch themes.

## Consequences

### Positive
- Sources panel is always visible — no click required to see citations
- Ambiguity is surfaced explicitly, not hidden in a single hedged answer
- Dark default matches the preference of the technical early adopter segment
- Three-panel scales well as features are added (e.g., document viewer in the Q&A area)

### Negative
- Three panels requires minimum ~900px width — not suitable for smaller windows
- Sources panel may feel redundant for very simple, high-confidence answers
- Implementing ambiguity detection requires the backend to return a structured
  `interpretations` field, not just a string answer
