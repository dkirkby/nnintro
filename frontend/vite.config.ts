import { defineConfig } from "vitest/config";

export default defineConfig({
  // Relative, so the built site works from a project page path.
  base: "./",
  test: {
    // Node by default: most tests here are numerical. A view test opts in
    // with a `@vitest-environment jsdom` docblock.
    environment: "node",
  },
});
