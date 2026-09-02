// Test entry point: `npm test` inside server/.
// Plain Node, no framework. Unit suites first, then the socket.io end-to-end run.

const { summary } = require('./harness');

const suites = [
  ['replayEngine', require('./replayEngine.test')],
  ['replayRecorder', require('./replayRecorder.test')],
  ['replayGithub', require('./replayGithub.test')],
  ['replayStore', require('./replayStore.test')],
  ['replay e2e', require('./replay.e2e')],
];

async function main() {
  const started = Date.now();
  for (const [name, suite] of suites) {
    try {
      await suite.run();
    } catch (err) {
      console.log(`\n  FATAL in ${name}: ${(err && err.stack) || err}`);
      process.exitCode = 1;
      break;
    }
  }

  const { passed, failed, failures } = summary();
  console.log(`\n════════════════════════════════════════`);
  console.log(`  ${passed} passed, ${failed} failed  (${Date.now() - started} ms)`);
  if (failed > 0) {
    console.log('  failing tests:');
    for (const failure of failures) console.log(`    - ${failure.suite} › ${failure.name}`);
  }
  console.log(`════════════════════════════════════════`);
  // The server keeps intervals alive, so exit explicitly.
  process.exit(failed > 0 || process.exitCode === 1 ? 1 : 0);
}

main();
