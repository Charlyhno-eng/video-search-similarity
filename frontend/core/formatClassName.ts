/**
 * Formats a class name to be lowercase and replaces non-alphanumeric characters with underscores.
 *
 * @param {string} name - The original class name.
 * @returns {string} The sanitized class name.
 */
export const formatClassName = (name: string): string => {
  return name.toLowerCase().replace(/[^a-z0-9]/g, "_");
};
