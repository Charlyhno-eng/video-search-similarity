/**
 * Converts a folder name into a more human-readable format.
 * Underscores are replaced with spaces, and the first letter is capitalized.
 * If the folder name is empty, returns "Root".
 *
 * @param folderName - The original folder name (e.g., "water_tree_reflect")
 * @returns A formatted string with spaces and capitalized first letter (e.g., "Water tree reflect")
 */
export function formatFolderName(folderName: string): string {
  if (!folderName) return "Root"; // fallback for empty strings
  // Replace underscores with spaces
  let formatted = folderName.replace(/_/g, " ");
  // Capitalize first letter
  formatted = formatted.charAt(0).toUpperCase() + formatted.slice(1);
  return formatted;
}
