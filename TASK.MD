# GameLens Frontend — Full Scale Implementation Guide

---

## Pre-Implementation Audit (Do This First)

Before writing a single line, you need to verify four things that will break everything downstream if ignored.

**Audit 1 — JSON field consistency**
Open `data/processed/transformed_df.csv` and `steam_games.json.gz` side by side. The recommendation API vocab was built from `transformed_df.csv` using the `item_name` column. Your frontend will use `app_name` from `steam_games.json.gz`. These two values must be identical for the same game, otherwise your similar games and recommendation rows will 404 silently. Run a spot check on 20 games manually — check `app_name` in the raw JSON against `item_name` in the CSV. If they differ, you need a mapping table.

**Audit 2 — Image filename verification**
Pick 10 game names from the JSON. Run the sanitize logic on each manually. Check that a file with that exact sanitized name exists in `steam_header/`. Pay specific attention to games with colons, apostrophes, dots, and non-ASCII characters — those are the most common failure points. Example risk cases: `"Hellblade: Senua's Sacrifice"` becomes `"Hellblade Senuas Sacrifice"` after sanitization — verify the scraper produced exactly that filename.

**Audit 3 — API health check**
Hit `GET http://localhost:8000/health` and confirm it returns `200`. Hit `GET http://localhost:8000/v1/recommendations?user_id=guest_test&count=5` and verify the shape of the response matches what `PersonalizedRow` will expect. Check that `recommendations[].item_name` is the field name — not `item_id`.

**Audit 4 — Docker network naming**
Run `docker network ls` and `docker inspect game-rec-api` to confirm what internal hostname the API container is reachable at. It should be `api` based on the service name in `docker-compose.yml`. Confirm this before writing the Next.js proxy config — if the service name is different, the proxy will silently fail.

---

## Repository Structure — Full Breakdown

The frontend lives as a sibling to your existing backend in the monorepo. The root of your project currently has `recommendation_api/`, `training/`, `pipelines/`, `data/`, `model_artifacts/`, `docker-compose.yml`. The frontend adds one new sibling folder:

```
GameLens/                               ← your existing repo root
│
├── docker-compose.yml                  ← will be extended
├── steam_header/                       ← your scraped images (already exists)
├── data/                               ← processed CSV and JSON (already exists)
├── model_artifacts/                    ← trained artifacts (already exists)
├── recommendation_api/                 ← your existing FastAPI app
├── training/                           ← your existing training pipeline
├── pipelines/                          ← your existing pipeline jobs
│
└── gamelens-web/                       ← NEW: entire Next.js frontend lives here
    │
    ├── Dockerfile
    ├── .dockerignore
    ├── next.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── package.json
    ├── .env.local                       ← local dev only, not committed
    │
    ├── public/
    │   ├── images/                      ← Docker volume mount point at runtime
    │   │                                   (empty at build time, filled by volume)
    │   └── placeholder.jpg              ← fallback image baked into build
    │
    ├── data/
    │   └── games.json                   ← built by prep script, copied in by Dockerfile
    │
    ├── scripts/
    │   └── build-games-json.js          ← one-time data prep script (Node, not bundled)
    │
    ├── lib/
    │   ├── games.js                     ← server-side JSON cache + pagination logic
    │   ├── sanitize.js                  ← filename sanitizer matching scraper logic
    │   └── user.js                      ← client-side anonymous user_id management
    │
    ├── hooks/
    │   ├── useGames.js                  ← paginated games fetcher with load-more state
    │   ├── useRecommendations.js        ← rec API fetcher with loading/error states
    │   └── useSimilarGames.js           ← item-to-item rec fetcher
    │
    ├── pages/
    │   ├── _app.js                      ← global layout, user_id init, event bus
    │   ├── _document.js                 ← HTML shell, fonts
    │   ├── index.js                     ← homepage
    │   ├── search.js                    ← search results
    │   │
    │   ├── game/
    │   │   └── [app_name].js            ← game detail page (static generation)
    │   │
    │   └── user/
    │       └── [user_id].js             ← user profile + personalized grid
    │
    ├── pages/api/
    │   ├── games.js                     ← GET /api/games?page=&genre=&q=
    │   ├── genres.js                    ← GET /api/genres
    │   └── game/
    │       └── [app_name].js            ← GET /api/game/:app_name
    │
    ├── components/
    │   │
    │   ├── layout/
    │   │   ├── Navbar.jsx
    │   │   ├── Footer.jsx
    │   │   └── PageWrapper.jsx          ← consistent max-width + padding shell
    │   │
    │   ├── game/
    │   │   ├── GameCard.jsx             ← THE most important component
    │   │   ├── GameGrid.jsx             ← grid container + load more orchestration
    │   │   ├── GameRow.jsx              ← horizontal scroll strip
    │   │   ├── GameDetailHero.jsx       ← large image + all metadata
    │   │   └── GameCardSkeleton.jsx     ← loading placeholder matching GameCard dimensions
    │   │
    │   ├── filters/
    │   │   ├── GenreFilter.jsx          ← pill chips, multi-select
    │   │   └── SearchBar.jsx            ← debounced input
    │   │
    │   ├── recommendations/
    │   │   ├── PersonalizedRow.jsx      ← homepage "For You" section
    │   │   └── SimilarGamesRow.jsx      ← detail page "More Like This" section
    │   │
    │   └── ui/
    │       ├── LoadMoreButton.jsx
    │       ├── SentimentBadge.jsx
    │       ├── PriceBadge.jsx
    │       ├── SpecTag.jsx
    │       └── ErrorState.jsx           ← generic error display with retry
    │
    └── styles/
        └── globals.css
```

---

## Step 1 — Data Preparation Script

**File: `gamelens-web/scripts/build-games-json.js`**

This script runs once, outside Docker, before you build the container. Its job is to transform `steam_games.json.gz` (the same file from your data ingestion) into a clean flat JSON array that the frontend can consume. It runs on your host machine, not inside any container.

**What it does, in order:**

First, it reads `steam_games.json.gz` from `../data/raw/steam_games.json.gz` (relative to the script location). It streams the gzip, reads line by line, parses each JSON object.

For each game entry it extracts and normalizes: `id`, `app_name` (fallback to `title` if missing), `genres` (ensure it's always an array, default to empty array), `tags` (same), `price` (normalize — `"Free to Play"` and `"Free To Play"` both become the string `"Free"`, numeric prices stay as numbers, missing becomes `null`), `release_date`, `sentiment` (missing becomes `null`), `specs` (array, default empty), `developer`, `publisher`.

It then runs the sanitize function on `app_name` to produce `image_filename`. It does NOT check whether the file exists — that check happens at render time. The `image_filename` field is what the frontend uses to construct the image path.

It sorts the output array by `app_name` alphabetically.

It writes the result to `gamelens-web/data/games.json`.

**Run it with:** `node gamelens-web/scripts/build-games-json.js`

**Verify it worked:** check the file exists, check the count matches what you expect, spot check 5 entries for the fields being correct.

**Critical:** the `app_name` in this output must exactly match the `item_name` in your recommendation API's vocab. If the audit from the pre-implementation step found mismatches, add a normalization step here to align them.

---

## Step 2 — Scaffold Next.js Project

**Run from inside `gamelens-web/`:**

Initialize with `npx create-next-app@latest . --no-app --no-src-dir --no-typescript --import-alias "@/*"`. You want Pages Router, not App Router — this keeps the structure simpler and aligns with the `pages/` layout described above. No TypeScript — keeps the barrier low.

Then install dependencies: `tailwindcss`, `postcss`, `autoprefixer` for styling. `swr` for data fetching in components — this gives you caching, revalidation, and loading states for free. `uuid` for generating anonymous user IDs.

Run `npx tailwindcss init -p` to generate the Tailwind and PostCSS config files.

**`tailwind.config.js` — what to configure:**
Set the `content` array to include `./pages/**/*.{js,jsx}` and `./components/**/*.{js,jsx}`. Set the theme to use a dark color palette as the default. Define a custom `steam-blue` color using `#1b2838` (Steam's navy) and `steam-teal` as `#66c0f4`. These will be your primary brand colors throughout the UI.

**`next.config.js` — what to configure:**
This file does two critical things. First, it sets up API rewrites so that any browser request to `/rec/*` gets proxied internally to `http://api:8000/*` inside Docker. This is what eliminates CORS — from the browser's perspective everything is same-origin. The rewrite rule maps `/rec/:path*` to `http://api:8000/:path*`. Second, for local development, use an environment variable `NEXT_PUBLIC_REC_API_URL` that defaults to `http://localhost:8000` when the Docker proxy isn't available. This means `useRecommendations` and similar hooks check the env var and use it directly in local dev, but use `/rec` prefix in Docker.

**`.env.local` — for local development only:**
Set `NEXT_PUBLIC_REC_API_URL=http://localhost:8000`. This file is never committed and never copied into Docker.

---

## Step 3 — `lib/sanitize.js`

This is two lines of logic but it is one of the most important files in the project. It must produce identical output to the Python scraper's `sanitize_filename` function.

**The logic:**
If the input is null or undefined, return the string `"__unknown__"`. Otherwise convert to string, apply a regex that removes all characters in the set `\ / * ? : " < > |`, then trim whitespace from both ends.

**Port this exactly from Python to JavaScript.** The regex character class is identical. The `.strip()` becomes `.trim()`.

**Test it manually** against 10 game names from your JSON before proceeding. Specifically test: a name with a colon, a name with quotes, a name with a forward slash, a name with Chinese/Japanese characters (these pass through unchanged — verify that's what the scraper did too), and a name that's just a number.

---

## Step 4 — `lib/games.js`

This is the server-side data layer. It only ever runs in Node.js — never in the browser. It uses a module-level variable as a cache so the file is read from disk exactly once per container lifetime.

**The cache variable:** declared at module scope, initialized to `null`.

**`loadGames()` function:** checks if cache is null. If null, reads `data/games.json` synchronously using Node's `fs.readFileSync` (synchronous is fine here — it only happens once), parses the JSON, stores in the cache variable, returns it. If not null, returns the cache directly.

**`getGames({ page, genre, q })` function:**
- Calls `loadGames()` to get the full array
- If `genre` is provided (non-empty string), filters to games where `game.genres` array includes that genre (case-insensitive comparison)
- If `q` is provided (non-empty string after trim), filters to games where `game.app_name` contains `q` (case-insensitive)
- Computes `total` = length of filtered array
- Computes `skip` = `(page - 1) * 10`
- Slices `filteredGames.slice(skip, skip + 10)`
- Returns `{ games: slice, total, page, hasMore: skip + 10 < total }`

**`getGame(app_name)` function:** calls `loadGames()`, finds the first entry where `game.app_name === app_name` (exact match), returns it or `null`.

**`getGenres()` function:** calls `loadGames()`, flattens all `genres` arrays, counts occurrences of each genre string, returns an array of `{ genre, count }` sorted by count descending.

**`getAllAppNames()` function:** calls `loadGames()`, returns just the `app_name` field from every entry. Used by `getStaticPaths` on the game detail page.

---

## Step 5 — `lib/user.js`

Client-side only. Never imported from server-side code.

**`getUserId()` function:** checks `typeof window !== 'undefined'` first — if false (SSR), returns `null`. If true, checks `localStorage.getItem('gamelens_uid')`. If it exists, returns it. If not, generates a new ID in the format `guest_` followed by a UUID v4, saves it to localStorage, returns it.

**`recordEvent(userId, itemName, eventType, playtime)` function:** constructs the event payload matching exactly what your `POST /v1/events` endpoint expects: `{ user_id, item_name, event_type, playtime, metadata: {} }`. POSTs to either `${process.env.NEXT_PUBLIC_REC_API_URL}/v1/events` (local) or `/rec/v1/events` (Docker). Fire-and-forget — no await, errors are silently caught. You don't want event tracking to ever break the UI.

---

## Step 6 — API Routes

### `pages/api/games.js`

Receives query params: `page` (default 1, parse as integer), `genre` (default empty string), `q` (default empty string).

Validates `page` is a positive integer. Calls `getGames({ page, genre, q })` from `lib/games.js`. Returns the result as JSON with status 200. On any error, returns status 500 with `{ error: "Failed to load games" }`.

This is the only endpoint the browser calls for the game grid. All filtering and pagination goes through here.

### `pages/api/genres.js`

No params. Calls `getGenres()` from `lib/games.js`. Returns the genre array. Add a cache header: `Cache-Control: public, max-age=3600` — genres don't change between deploys.

### `pages/api/game/[app_name].js`

Receives `app_name` from the dynamic route. Calls `getGame(app_name)` from `lib/games.js`. If null, returns 404. Otherwise returns the game object. This route is used by the game detail page for client-side data needs (mostly it uses static generation, but this is the fallback).

---

## Step 7 — Custom Hooks

### `hooks/useGames.js`

This hook manages all state for the game grid. It owns: `games` array, `page` number, `hasMore` boolean, `loading` boolean (initial load), `loadingMore` boolean (load more clicks), `error`.

**State flow:**
- On mount, or whenever `genre` or `q` props change: reset `games` to empty array, reset `page` to 1, set `loading` to true, fetch `/api/games?page=1&genre=X&q=Y`, on success set `games` to result and `hasMore`, set `loading` false
- `loadMore()` function: if `loadingMore` is true or `hasMore` is false, return early. Set `loadingMore` true, fetch next page, on success append results to `games` array, increment `page`, update `hasMore`, set `loadingMore` false.

Use `useEffect` with `[genre, q]` dependency array for the reset-and-reload behavior. Use a separate `loadMore` callback with `useCallback`.

**Abort controller:** inside the `useEffect`, create an `AbortController`. Pass its signal to the fetch call. In the cleanup function, call `abort()`. This prevents state updates on unmounted components when genre changes rapidly.

### `hooks/useRecommendations.js`

Takes `userId` and `count` (default 10) as params.

On mount (and when `userId` changes from null to a real value), fetches from the rec API: `GET /rec/v1/recommendations?user_id=${userId}&count=${count}` (Docker) or `${NEXT_PUBLIC_REC_API_URL}/v1/recommendations?...` (local).

State: `recommendations` array (default empty), `loading` boolean, `error`.

If `userId` is null, do nothing — return empty state. The component handles the null case.

On success, maps the response `recommendations` array. Each item from the API has `item_name` and `score`. You need to enrich these with display data (genres, price, image filename). Do this by fetching `/api/game/${item_name}` for each recommendation OR by passing the full games list down as a prop and doing a client-side lookup. The second approach is faster — pass the full enriched game data from `_app.js` context and do a lookup by `app_name === item_name`.

### `hooks/useSimilarGames.js`

Takes `appName` as param. Fetches `GET /rec/v1/items/${encodeURIComponent(appName)}/similar?count=10`. Returns `{ similar, loading, error }`. Same enrichment pattern as `useRecommendations` — look up game data by name from context.

---

## Step 8 — Image Handling Strategy (Placeholder System)

This is the answer to your requirement about working without images. Every place an image is rendered follows a three-tier fallback:

**Tier 1 — Local volume image:** `/images/${sanitize(app_name)}.jpg` — this is the scraped header image from the Docker volume mount.

**Tier 2 — Steam CDN fallback:** `https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${id}/header.jpg` — this works if the container has internet access and the Steam CDN is reachable.

**Tier 3 — Placeholder:** `/placeholder.jpg` — a generic image baked into the build under `public/placeholder.jpg`. This should be a dark grey rectangle with a game controller icon or just the GameLens logo. Dimensions should match Steam header image aspect ratio which is 460×215 pixels (roughly 2.14:1 ratio).

**How to implement this in `GameCard.jsx`:**
Use a regular HTML `<img>` tag (not `next/image` — volumes aren't compatible with Next.js image optimization). Set the `src` to Tier 1. Add an `onError` handler. The `onError` handler checks a `fallbackLevel` state variable (starts at 0). On first error, increment to 1 and set `src` to the CDN URL. On second error, increment to 2 and set `src` to `/placeholder.jpg`. On third error (placeholder itself failing), set a boolean `showInitials` state to true and render a CSS-only placeholder instead of an `<img>` at all.

**The CSS-only placeholder:** a `div` with the same aspect ratio as the image, background color `#1b2838`, centered text showing the first two characters of the game name in large white letters. This is the ultimate fallback and will always work.

**Why this matters:** During development before the volume is mounted, all 8,649 images will fail Tier 1. With this system, they silently fall to CDN or placeholder — the UI is never broken, just potentially slower on CDN loads.

---

## Step 9 — Core Components

### `components/ui/GameCardSkeleton.jsx`

Renders a card-shaped div with the same dimensions as a real `GameCard`. Uses Tailwind's `animate-pulse` class. Contains: a rectangle at the top (image area, same aspect ratio as headers), two shorter rectangles below (title and genres lines). Background colors use `bg-gray-700` and `bg-gray-600` to match the dark theme. No props needed.

### `components/game/GameCard.jsx`

This component does the most work. Props: `game` object (full game data from `games.json` shape), `onClick` callback.

**Renders:**
- Outer `div` — `cursor-pointer`, hover scale transform (`hover:scale-102`), dark background, rounded corners, overflow hidden
- Image area — aspect ratio wrapper div (padding-bottom trick: `pb-[46.7%]` for 460×215 ratio), contains the `<img>` with the three-tier fallback system described in Step 8. The image uses `object-cover` so it fills the space regardless of actual image dimensions.
- Content area — below the image: game title (one line, truncate overflow), genre chips (first two genres only, rendered as small colored pills), price badge
- On click: calls `onClick` if provided, and also calls `recordEvent(userId, game.app_name, 'click', 0)` from `lib/user.js`

**Important:** the `userId` comes from a React context set up in `_app.js`, not from props. Don't thread `userId` through every component — use context.

### `components/game/GameGrid.jsx`

Props: `genre` (string), `q` (string from search).

Uses `useGames(genre, q)` hook internally.

Renders:
- If `loading` is true: a grid of 10 `GameCardSkeleton` components — same layout as real cards
- If `error` is set: the `ErrorState` component with a retry button that calls the hook's `reload` function
- If `games` is empty after loading: a centered "No games found" message with a suggestion to clear filters
- Otherwise: a responsive CSS grid (`grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5`) of `GameCard` components, followed by the `LoadMoreButton` component

Each `GameCard` receives a click handler that navigates to `/game/${encodeURIComponent(game.app_name)}`.

### `components/game/GameRow.jsx`

Props: `title` (section heading string), `games` array, `loading` boolean, `onCardClick`.

Renders:
- Section title in `h2` style
- A horizontally scrollable `div` with `overflow-x-auto`, `flex`, `gap-4`, `pb-4` (space for scrollbar), `scrollbar-thin` styling
- Each game is a narrower `GameCard` variant — same component but smaller, width fixed at `w-48`
- If `loading`: 8 `GameCardSkeleton` components in a row
- If `games` is empty: nothing — the parent section handles empty state

### `components/filters/GenreFilter.jsx`

Props: `selectedGenre` (string or null), `onGenreChange` (callback).

Fetches genres from `/api/genres` using SWR on mount. SWR caches this for the session.

Renders a horizontally scrollable row of pill buttons. One pill for "All" (clears selection). One pill per genre from the API response. Active genre has a highlighted background (`bg-steam-teal text-black`). Inactive genres have dark background with border.

On click: if the clicked genre is already selected, call `onGenreChange(null)` (deselect). Otherwise call `onGenreChange(genre)`.

### `components/filters/SearchBar.jsx`

Props: `onSearch` callback.

Internal state: `inputValue` (what the user is typing).

Uses `useEffect` with a 300ms debounce: when `inputValue` changes, wait 300ms, then call `onSearch(inputValue.trim())`. Clear the timeout on each change. This prevents an API call on every keystroke.

Renders a styled text input with a search icon. Clear button appears when `inputValue` is non-empty — clicking it sets `inputValue` to empty and calls `onSearch("")`.

### `components/recommendations/PersonalizedRow.jsx`

No props needed. Gets `userId` from context.

Uses `useRecommendations(userId, 10)` hook.

Renders a `GameRow` with title "Recommended For You" if userId is a real guest ID, or "Popular Right Now" if userId is null (SSR state). The title switches on the client after hydration.

If the rec API is unreachable (network error), renders nothing — the homepage doesn't break, this section just silently disappears. Log the error to console only.

### `components/recommendations/SimilarGamesRow.jsx`

Props: `appName`.

Uses `useSimilarGames(appName)` hook.

Renders a `GameRow` with title "More Like This". If loading: skeleton row. If error or empty: renders nothing (same silent failure policy).

### `components/ui/LoadMoreButton.jsx`

Props: `onClick`, `loading` (boolean), `hasMore` (boolean), `total`, `loaded`.

If `hasMore` is false: renders a small centered text like "Showing all ${total} games" — no button.

If `hasMore` is true and `loading` is false: renders a centered button "Load 10 more (${total - loaded} remaining)".

If `loading` is true: renders the button in disabled state with a spinner icon.

---

## Step 10 — `pages/_app.js`

This is the global setup file. It does three things:

**Thing 1 — UserID initialization:**
A `useEffect` with empty dependency array (runs once on mount, client-side only) calls `getUserId()` from `lib/user.js` and stores the result in a `useState` variable. This is then provided via React Context to the entire app. All components that need the `userId` (GameCard for event tracking, PersonalizedRow for fetching) get it from this context, not from props.

**Thing 2 — Global layout:**
Wraps every page in `PageWrapper` which contains `Navbar` and `Footer`. Passes `router.pathname` to `Navbar` so it can highlight the active section.

**Thing 3 — Games context (optional optimization):**
If you want to avoid per-recommendation API calls to enrich items, load a lightweight name-to-metadata map here using SWR on a `/api/games?all=true` route (you'd add an `all` param to the API route that bypasses pagination). This map lets hooks do instant local lookups by `app_name` to get genres, price, image path for recommendation results. This avoids N+1 fetches when showing 10 recommendations.

---

## Step 11 — `pages/index.js` (Homepage)

State managed at this page level: `selectedGenre` (null initially), `searchQuery` (empty string initially).

**Page structure from top to bottom:**

**Section 1 — Hero bar:**
Full-width dark bar with the GameLens logo on the left, `SearchBar` in the center taking up 40% width, a "My Profile" link on the right (links to `/user/${userId}` — filled in after hydration, shows nothing before). Background is `steam-blue`.

**Section 2 — Personalized row:**
`PersonalizedRow` component. Full width. Shows the horizontal scroll of recommended games. Loads independently of the grid below.

**Section 3 — Filter + Grid:**
`GenreFilter` on top, then `GameGrid` below it. `GenreFilter`'s `onGenreChange` updates `selectedGenre`. `SearchBar`'s `onSearch` updates `searchQuery`. Both state variables are passed as props to `GameGrid`. When either changes, `GameGrid`'s `useGames` hook resets and reloads.

**No SSR data fetching on the homepage.** The page shell is static. The `GameGrid` fetches client-side. This means the first paint is fast (just the shell), and data appears after. The skeleton state handles the loading period.

---

## Step 12 — `pages/game/[app_name].js`

**`getStaticPaths`:** calls `getAllAppNames()` from `lib/games.js`, returns all names as paths. Set `fallback: 'blocking'` — if a game was added after build time, Next.js will generate the page on first request and cache it.

**`getStaticProps`:** receives `params.app_name`, calls `getGame(app_name)`, returns it as props. If null, returns `{ notFound: true }` — Next.js shows the 404 page automatically.

**Page structure:**

**Top section — `GameDetailHero`:**
Large header image using the same three-tier fallback system. Image displayed at full width up to a max-width container. Below it: game title in large heading, genre pills, price, sentiment badge, release date, developer name.

**Middle section — description/specs:**
Specs listed as tags (`Steam Achievements`, `Multiplayer`, etc.). These come from the `specs` array in the game data.

**Bottom section — `SimilarGamesRow`:**
Passes `appName` to the component. Loads asynchronously — page renders immediately, this section fills in after. If the recommendation API is down, this section just doesn't appear. No error shown to the user — it's supplementary content.

**On mount side effect:**
`useEffect` fires `recordEvent(userId, app_name, 'impression', 0)` once. This feeds into your NearlineUpdater and starts personalizing the homepage for this user.

---

## Step 13 — `pages/search.js`

Reads `q` from `router.query` on mount.

Pre-populates the `SearchBar` with the query string from the URL.

Uses `useGames(null, queryFromUrl)` to show filtered results.

The `GenreFilter` is also present here — selecting a genre narrows the already-searched results further.

URL stays in sync: when search or genre changes, push to `router` with `router.push('/search?q=X&genre=Y')` so the URL is shareable and the back button works correctly.

---

## Step 14 — `pages/user/[user_id].js`

No static generation here — user pages are server-rendered or client-rendered only.

On mount: if `router.query.user_id` doesn't match `getUserId()`, show a "This is not your profile" message. If it does match, proceed.

Shows the personalized grid using `useRecommendations(userId, 20)` with a larger count. Below that, shows a list of played games fetched from `GET /rec/v1/users/${userId}/played` — however this endpoint doesn't exist on your current API. For now, this section can be skipped or shown as a placeholder. Mark it as a "coming soon" section visually.

---

## Step 15 — `components/layout/Navbar.jsx`

Contains:
- Logo/brand name on the left, links to `/`
- Search input (smaller version of SearchBar) in the middle — on submit navigates to `/search?q=X`
- Right side: shows "Guest" with a user icon before hydration, switches to a truncated user ID link after hydration (e.g., "guest_4f2a..." linking to `/user/guest_4f2a...`)

The right side uses `suppressHydrationWarning` on the element that shows the user ID, since it changes between server render (null) and client render (real ID). Without this, React will log a hydration mismatch warning.

---

## Step 16 — Dockerfile for the Web Container

**Build stage:**
Uses `node:20-alpine` base image. Sets working directory to `/app`. Copies `package.json` and installs dependencies. Copies the entire `gamelens-web/` directory. Copies `data/games.json` from the build context into `/app/data/games.json`. Runs `next build`.

**Run stage:**
Uses `node:20-alpine` again. Copies the built output from the build stage. Exposes port 3000. Command is `node_modules/.bin/next start`.

**`.dockerignore` for the web container:**
Excludes `node_modules`, `.next`, `.env.local`, `public/images/*` (images come via volume, not build). The `public/images/` directory itself should exist but be empty — Docker volumes mount into it at runtime.

**Important note on the volume mount:** the `public/images/` directory must exist in the image even if empty. If it doesn't exist, Docker creates it as a root-owned directory and the Next.js static server may have permission issues. Create it explicitly in the Dockerfile with `RUN mkdir -p /app/public/images`.

---

## Step 17 — Updated `docker-compose.yml`

Adding the `web` service to your existing file:

**Redis service:** no changes.

**API service:** add a `healthcheck` block that tests `GET http://localhost:8000/health` every 10 seconds, with a 5 second timeout and 3 retries. This is what the `web` service waits for.

**New `web` service:**
- Build context: `./gamelens-web`
- Dockerfile: `./gamelens-web/Dockerfile`
- Container name: `game-rec-web`
- Port: `3000:3000`
- Environment variables: `NEXT_PUBLIC_REC_API_URL` is not needed in Docker (proxy handles it), but set `NODE_ENV=production`
- Volumes: `./steam_header:/app/public/images:ro` — read-only mount of your scraped images folder directly into the Next.js static serving path
- `depends_on`: `api` with `condition: service_healthy`
- `restart: unless-stopped`

**`next.config.js` rewrite rule:**
The proxy destination should be `http://api:8000` — using the Docker service name `api` which resolves to the API container's internal IP on the Docker bridge network. The rewrite maps `/rec/:path*` to `http://api:8000/:path*`.

---

## Step 18 — Integration Verification Sequence

Run these in order after `docker-compose up --build`:

**Check 1:** `docker logs game-rec-web` — should show Next.js build output ending in "ready on port 3000"

**Check 2:** `curl http://localhost:3000/api/genres` — should return JSON array of genre objects. If this fails, `lib/games.js` can't read the JSON file. Check that `data/games.json` was copied correctly in the Dockerfile.

**Check 3:** `curl http://localhost:3000/api/games?page=1` — should return 10 games with `hasMore` and `total` fields.

**Check 4:** `curl http://localhost:3000/rec/v1/health` — this tests the proxy. Should return the same response as `curl http://localhost:8000/health`. If it fails, the Docker network name is wrong — run `docker inspect game-rec-api` and check the network hostname.

**Check 5:** Open `http://localhost:3000` in browser. Open DevTools Network tab. Filter for `api/games` — should see a successful request with 10 items. Filter for `/rec/` — should see a recommendations request. Check console for errors.

**Check 6:** Click one game card. Should navigate to `/game/[name]`. Check Network tab for a `/rec/v1/items/[name]/similar` request. If 404, the name from `games.json` doesn't match the API vocab — go back to the pre-implementation audit.

**Check 7:** Check an image loads. Open one GameCard, inspect the `img` tag. If the `src` shows `/images/Counter-Strike.jpg` but the image is broken, the volume mount worked but the filename doesn't match. Run `ls /app/public/images | head -20` inside the container to see actual filenames versus expected.

---

## Known Integration Risk Points Summary

**Risk 1 — `app_name` vs `item_name` mismatch:** mitigated by Step 1 audit and the `_resolve_name` fuzzy lookup already added to the API in a previous session.

**Risk 2 — Image filename mismatch:** mitigated by the three-tier fallback and Step 8's CSS placeholder as the final layer.

**Risk 3 — Docker volume mount path:** `steam_header` in your repo root contains files like `Counter-Strike.jpg`. The volume mount maps it to `/app/public/images`. Next.js serves `/app/public/` as the static root, so the URL path becomes `/images/Counter-Strike.jpg`. Verify this end-to-end in Check 7.

**Risk 4 — Redis empty on web container startup:** the API's lifespan function now auto-populates Redis if empty (from the previous session's fix). The web container depends on `api` being healthy, which means the health check passes only after the lifespan function completes. Redis should already be populated by the time the web container starts making rec API calls.

**Risk 5 — Hydration mismatch on user ID:** the user ID comes from localStorage which doesn't exist during SSR. Any component that renders the user ID must use `suppressHydrationWarning` or render only after mounting via a `useEffect` + state pattern.