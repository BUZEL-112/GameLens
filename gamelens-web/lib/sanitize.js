/**
 * Sanitize a game name to produce the image filename.
 * Must produce identical output to the Python scraper's sanitize_filename().
 *
 * Logic: remove all characters in the set \ / * ? : " < > |
 * then trim whitespace from both ends.
 */
export function sanitize(name) {
  if (name == null) return "__unknown__";
  return String(name)
    .replace(/[\\/*?:"<>|'%]/g, "") // Added % to the list
    .trim();
}