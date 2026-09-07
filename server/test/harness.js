// Minimal test harness — plain Node, no framework.

const state = {
  passed: 0,
  failed: 0,
  failures: [],
  suite: '',
};

function suite(name) {
  state.suite = name;
  console.log(`\n── ${name} ──`);
}

async function test(name, fn) {
  try {
    await fn();
    state.passed++;
    console.log(`  ok    ${name}`);
  } catch (err) {
    state.failed++;
    state.failures.push({ suite: state.suite, name, err });
    console.log(`  FAIL  ${name}`);
    console.log(`        ${(err && err.stack) || err}`);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

// Key-order independent comparison.
function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).sort();
    return `{${keys.map(key => `${JSON.stringify(key)}:${stable(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function assertEqual(actual, expected, message) {
  if (stable(actual) !== stable(expected)) {
    throw new Error(`${message || 'values differ'}\n        actual:   ${stable(actual)}\n        expected: ${stable(expected)}`);
  }
}

async function assertThrows(fn, check, message) {
  let thrown = null;
  try {
    await fn();
  } catch (err) {
    thrown = err;
  }
  if (!thrown) throw new Error(message || 'expected an error but none was thrown');
  if (check) check(thrown);
  return thrown;
}

function summary() {
  return { passed: state.passed, failed: state.failed, failures: state.failures };
}

module.exports = { suite, test, assert, assertEqual, assertThrows, summary };
