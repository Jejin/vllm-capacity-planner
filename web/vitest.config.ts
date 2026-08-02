// The rendered page is its own failure surface. Two labels once described the flat derivation
// while displaying P50/P95 values, and 209 passing domain tests had nothing to say about it —
// the arithmetic was right and the sentence next to it was wrong.
//
// jsdom rather than a headless browser deliberately: these assertions are about TEXT, not
// pixels, so they need no fonts, no baseline images and no browser in CI.
import { defineConfig, type Plugin } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

/**
 * Strip `<style>` before vite-plugin-svelte sees it.
 *
 * The plugin routes every style block through vite's `preprocessCSS`, which builds an
 * environment that is not initialised under vitest — it throws "Cannot create proxy with a
 * non-object as target" before a single test collects. Nothing here asserts on CSS, so removing
 * the block is cheaper and clearer than fighting the pipeline.
 *
 * The consequence is worth stating plainly: these tests see text and structure and are blind to
 * styling. A rule that visually hid an element would not fail them.
 */
function stripStyles(): Plugin {
  return {
    name: 'strip-svelte-styles',
    enforce: 'pre',
    transform(code, id) {
      if (!id.endsWith('.svelte')) return null;
      return { code: code.replace(/<style>[\s\S]*?<\/style>/g, ''), map: null };
    },
  };
}

export default defineConfig({
  plugins: [stripStyles(), svelte({ hot: false, emitCss: false })],
  resolve: { conditions: ['browser'] },
  test: { environment: 'jsdom', css: false, include: ['src/__tests__/**/*.test.ts'] },
});
