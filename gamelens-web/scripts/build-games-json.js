/**
 * build-games-json.js
 *
 * One-time data preparation script. Runs on the host machine (not in Docker).
 * Reads steam_games.json.gz and produces a clean JSON array at data/games.json.
 *
 * Usage:
 *   node gamelens-web/scripts/build-games-json.js
 */

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const readline = require("readline");

const RAW_PATH = path.resolve(__dirname, "../../data/raw/steam_games.json.gz");
const OUTPUT_PATH = path.resolve(__dirname, "../data/games.json");

/**
 * Sanitize a game name to produce the image filename.
 * Mirrors the Python scraper's sanitize_filename() exactly.
 */
function sanitize(name) {
  if (name == null) return "__unknown__";
  return String(name)
    .replace(/[\\/*?:"<>|]/g, "")
    .trim();
}

/**
 * Normalize price values into a consistent format.
 */
function normalizePrice(price) {
  if (price == null || price === "") return null;
  const s = String(price).trim().toLowerCase();
  if (s === "free to play" || s === "free") return "Free";
  const num = parseFloat(s.replace(/[$,]/g, ""));
  if (!isNaN(num)) return num;
  return s;
}

function ensureArray(val) {
  if (Array.isArray(val)) return val;
  if (val == null) return [];
  if (typeof val === "string") return val.split(",").map((s) => s.trim()).filter(Boolean);
  return [];
}

async function main() {
  if (!fs.existsSync(RAW_PATH)) {
    console.error(`[ERROR] Raw data file not found at: ${RAW_PATH}`);
    console.error("Make sure steam_games.json.gz exists in data/raw/");
    process.exit(1);
  }

  console.log(`Reading ${RAW_PATH}...`);

  const gunzip = zlib.createGunzip();
  const stream = fs.createReadStream(RAW_PATH).pipe(gunzip);
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  const games = [];
  let lineCount = 0;
  let errorCount = 0;

  for await (const line of rl) {
    lineCount++;
    const trimmed = line.trim();
    if (!trimmed) continue;

    try {
      const entry = JSON.parse(trimmed);

      const appName = entry.app_name || entry.title || null;
      if (!appName) {
        errorCount++;
        continue;
      }

      games.push({
        id: entry.id || null,
        app_name: appName,
        genres: ensureArray(entry.genres),
        tags: ensureArray(entry.tags),
        price: normalizePrice(entry.price),
        release_date: entry.release_date || null,
        sentiment: entry.sentiment || null,
        specs: ensureArray(entry.specs),
        developer: entry.developer || null,
        publisher: entry.publisher || null,
        image_filename: sanitize(appName),
      });
    } catch (err) {
      errorCount++;
    }
  }

  // Sort alphabetically by app_name
  games.sort((a, b) => a.app_name.localeCompare(b.app_name));

  // Ensure output directory exists
  const outDir = path.dirname(OUTPUT_PATH);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(games, null, 0));

  console.log(`Done.`);
  console.log(`  Lines read:    ${lineCount}`);
  console.log(`  Games written: ${games.length}`);
  console.log(`  Errors:        ${errorCount}`);
  console.log(`  Output:        ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
