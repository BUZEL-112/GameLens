/**
 * Server-side game data layer.
 * Reads data/games.json once and caches in memory for the process lifetime.
 * Never import this file from client-side code.
 */
import fs from "fs";
import path from "path";

const PAGE_SIZE = 10;

let cache = null;

function loadGames() {
  if (cache) return cache;

  const filePath = path.join(process.cwd(), "data", "games.json");

  if (!fs.existsSync(filePath)) {
    console.warn("[games.js] data/games.json not found -- returning empty dataset");
    cache = [];
    return cache;
  }

  const raw = fs.readFileSync(filePath, "utf-8");
  cache = JSON.parse(raw);
  return cache;
}

/**
 * Paginated, filterable game listing.
 * @param {{ page?: number, genre?: string, q?: string }} opts
 */
export function getGames({ page = 1, genre = "", q = "" } = {}) {
  let games = loadGames();

  if (genre) {
    const g = genre.toLowerCase();
    games = games.filter(
      (game) =>
        Array.isArray(game.genres) &&
        game.genres.some((gn) => gn.toLowerCase() === g)
    );
  }

  if (q && q.trim()) {
    const query = q.trim().toLowerCase();
    games = games.filter(
      (game) =>
        game.app_name && game.app_name.toLowerCase().includes(query)
    );
  }

  const total = games.length;
  const skip = (page - 1) * PAGE_SIZE;
  const slice = games.slice(skip, skip + PAGE_SIZE);

  return {
    games: slice,
    total,
    page,
    hasMore: skip + PAGE_SIZE < total,
  };
}

/**
 * Retrieve all games (no pagination) -- used for recommendation enrichment.
 */
export function getAllGames() {
  return loadGames();
}

/**
 * Find a single game by exact app_name match.
 */
export function getGame(appName) {
  const games = loadGames();
  return games.find((g) => g.app_name === appName) || null;
}

/**
 * Aggregate genres across all games, sorted by frequency.
 * @returns {{ genre: string, count: number }[]}
 */
export function getGenres() {
  const games = loadGames();
  const counts = {};

  for (const game of games) {
    if (!Array.isArray(game.genres)) continue;
    for (const genre of game.genres) {
      counts[genre] = (counts[genre] || 0) + 1;
    }
  }

  return Object.entries(counts)
    .map(([genre, count]) => ({ genre, count }))
    .sort((a, b) => b.count - a.count);
}

/**
 * All app_name values -- used by getStaticPaths.
 */
export function getAllAppNames() {
  return loadGames().map((g) => g.app_name);
}
