import { useState, useEffect } from "react";

/**
 * Fetch item-to-item similar games for a given app name.
 */
export function useSimilarGames(appName) {
  const [similar, setSimilar] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!appName) return;

    setLoading(true);
    setError(null);

    const base = process.env.NEXT_PUBLIC_REC_API_URL || "/rec";
    const url = `${base}/v1/items/${encodeURIComponent(appName)}/similar?count=10`;

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setSimilar(data.similar_items || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error("[useSimilarGames]", err.message);
        setError(err.message);
        setLoading(false);
      });
  }, [appName]);

  return { similar, loading, error };
}
