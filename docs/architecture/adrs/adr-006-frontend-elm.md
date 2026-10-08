# ADR-006: Elm + elm-css for Frontend

## Status

Amended by [[adr-009-native-mac-app|ADR-009]] (2026-03-19)

Elm is retained as the frontend language. The change: HTTP calls replaced with Tauri IPC ports.
All reasoning for choosing Elm over React/Svelte remains valid.

## Date

2025-03-17

## Context

The original product plan specified Next.js or SvelteKit for the frontend. The team has decided to use Elm with elm-css instead.

The frontend is deferred until the citation engine PoC is validated, but this decision should be recorded now to inform future architecture choices (API design, data serialization format, etc.).

## Decision

**Use Elm with elm-css for all frontend development.**

### Why Elm
- **Compiler-guaranteed no runtime exceptions** — critical for a professional tool where crashes erode trust
- **Forced explicitness** — every state, every edge case must be handled. Aligns with our explainability-first philosophy
- **Predictable refactoring** — the Elm compiler catches every downstream effect of a change
- **Excellent for complex UI state** — evidence reports with expandable claims, verification details, and document cross-references involve significant state management
- **Team expertise** — the team has production Elm experience from healthcare applications

### Why elm-css
- Type-safe CSS that compiles with the rest of the Elm code
- No CSS class name typos, no unused styles
- Composable style functions

### API Design Implications
- Backend API should return well-structured JSON that maps cleanly to Elm decoders
- The evidence report data model (claims, citations, confidence scores) should be designed with JSON serialization in mind from the start
- Consider JSON:API or a similarly structured format that Elm can decode predictably

## Alternatives Considered

### Next.js 15 (App Router)
Rejected. Large ecosystem but brings significant JavaScript runtime complexity. Server components add architectural weight. React's runtime error model (things can crash in production) doesn't align with the reliability expectations of a legal tool.

### SvelteKit
Considered. Simpler than Next.js, good DX. But still JavaScript — no compiler-level safety guarantees. Smaller ecosystem than React for complex components.

### HTMX + server-rendered templates
Considered for simplicity. Could work for basic interactions but evidence reports with expandable claims, interactive verification details, and document navigation need rich client-side state management.

## Consequences

### Positive
- Zero runtime exceptions in production
- Confidence in refactoring as the UI evolves
- Clean separation of frontend and backend (Elm is always an SPA consuming an API)
- Type-safe data flow from API response through to rendered UI
- Aligns with the project's philosophy of reliability and transparency

### Negative
- Smaller talent pool than React/Next.js for future hiring
- Elm ecosystem is smaller — fewer off-the-shelf components
- Some JavaScript interop needed for browser APIs (ports)
- Learning curve for contributors unfamiliar with Elm
- May need custom components for things like PDF viewers, rich text editors

### Deferred Decisions
- Specific Elm packages for HTTP, routing, etc.
- Whether to use elm-ui in addition to or instead of elm-css
- How to handle document viewing/annotation in the browser
- Authentication flow (likely ports to a JS auth library)
