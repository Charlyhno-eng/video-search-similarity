/**
 * Formats a sanitized class name for display.
 * Converts underscores to spaces and capitalizes the first letter.
 *
 * @param {string} name - The sanitized class name (lowercase with underscores).
 * @returns {string} A human-readable string with spaces and first letter capitalized.
 */
export const formatClassName= (name: string): string => {
  if (!name) return "Root";
  const formatted = name.replace(/_/g, " ");
  return formatted.charAt(0).toUpperCase() + formatted.slice(1);
};
