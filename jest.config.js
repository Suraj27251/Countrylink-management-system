/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'jsdom',
  testPathIgnorePatterns: [
    '/node_modules/',
    // Playwright-based tests — run via `npm run test:playwright`
    'tests/tracking-page/no-overflow.property.test.js',
    'tests/tracking-page/min-font-size.property.test.js',
    'tests/tracking-page/integration.test.js',
  ],
};
