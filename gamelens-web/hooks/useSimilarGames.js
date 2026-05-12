import { useState, useEffect } from "react";

export function useSimilarGames(appName) {
  const [similar, setSimilar] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!appName) return;

    setLoading(true);
    setError(null);

    const base = process.env.NEXT_PUBLIC_REC_API_URL || "/rec";

    Promise.all([
      fetch(`${base}/v1/items/${encodeURIComponent(appName)}/similar?count=10`)
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
      fetch(`/api/games?all=true`)
        .then(r => r.json())
    ])
      .then(([simData, gamesData]) => {
        const lookup = {};
        for (const g of gamesData.games || []) lookup[g.app_name] = g;

        const enriched = (simData.similar_items || [])
          .map(s => lookup[s.item_name])
          .filter(Boolean);

        setSimilar(enriched);
        setLoading(false);
      })
      .catch(err => {
        console.error("[useSimilarGames]", err.message);
        setError(err.message);
        setLoading(false);
      });
  }, [appName]);

  return { similar, loading, error };
}