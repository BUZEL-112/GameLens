import { getGame } from "@/lib/games";

export default function handler(req, res) {
  try {
    const { app_name } = req.query;
    const game = getGame(decodeURIComponent(app_name));

    if (!game) {
      return res.status(404).json({ error: "Game not found" });
    }

    return res.status(200).json(game);
  } catch (err) {
    console.error("[/api/game]", err);
    return res.status(500).json({ error: "Failed to load game" });
  }
}
