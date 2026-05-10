import { getGenres } from "@/lib/games";

export default function handler(req, res) {
  try {
    res.setHeader("Cache-Control", "public, max-age=3600");
    const genres = getGenres();
    return res.status(200).json(genres);
  } catch (err) {
    console.error("[/api/genres]", err);
    return res.status(500).json({ error: "Failed to load genres" });
  }
}
