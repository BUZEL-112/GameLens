/**
 * build-games-json.js
 *
 * One-time data preparation script.
 * Reads steam_games.json and produces a clean JSON array at data/games.json.
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");

const RAW_PATH = path.resolve(__dirname, "../../data/raw/steam_games.json");
const OUTPUT_PATH = path.resolve(__dirname, "../data/games.json");

/**
 * Sanitize a game name to produce the image filename.
 * Removes characters that are illegal in filenames or specifically 
 * excluded in your sample (like apostrophes).
 */
function sanitize(name) {
  if (name == null) return "__unknown__";
  return String(name)
    .replace(/[\\/*?:"<>|'%]/g, "") // Added % to the list
    .trim();
}

/**
 * Normalize price values into a consistent format.
 */
function normalizePrice(price) {
  if (price == null || price === "") return "Free";
  const s = String(price).trim().toLowerCase();
  
  // Handle various "Free" strings
  if (["free to play", "free", "free demo", "play for free"].includes(s)) {
    return "Free";
  }

  // Extract numeric value
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
    process.exit(1);
  }

  console.log(`Reading ${RAW_PATH}...`);

  const stream = fs.createReadStream(RAW_PATH);
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

      // Your sample output requires an app_name and an id
      const appName = entry.app_name || entry.title || null;
      if (!appName || !entry.id) {
        continue; 
      }

      // Construct the object in the exact order of your sample
      games.push({
        id: String(entry.id),
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

  // Sort by app_name to match your desired structure
  games.sort((a, b) => a.app_name.localeCompare(b.app_name));

  // Ensure output directory exists
  const outDir = path.dirname(OUTPUT_PATH);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  // Use null, 2 to create the pretty-printed array format you requested
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(games, null, 2));

  console.log(`Done.`);
  console.log(`  Lines read:     ${lineCount}`);
  console.log(`  Games written:  ${games.length}`);
  console.log(`  Errors:         ${errorCount}`);
  console.log(`  Output:         ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});