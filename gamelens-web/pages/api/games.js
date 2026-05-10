import { getGames, getAllGames } from "@/lib/games";

export default function handler(req, res) {
  try {
    const { page, genre, q, all } = req.query;

    // Optional: return all games for client-side enrichment
    if (all === "true") {
      const games = getAllGames();
      return res.status(200).json({ games, total: games.length });
    }

    const pageNum = Math.max(1, parseInt(page, 10) || 1);
    const result = getGames({
      page: pageNum,
      genre: genre || "",
      q: q || "",
    });

    return res.status(200).json(result);
  } catch (err) {
    console.error("[/api/games]", err);
    return res.status(500).json({ error: "Failed to load games" });
  }
}
