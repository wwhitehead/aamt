// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['js/packages/*/test/**/*.test.ts'],
    // The tree and stream suites check hundreds of thousands of points; give them room on CI runners.
    testTimeout: 120_000,
  },
});
