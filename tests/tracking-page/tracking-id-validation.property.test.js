/**
 * Property-Based Test: Tracking ID Validation (Property 4)
 *
 * Property 4: Tracking ID validation is correct
 * For any string input, the validation function SHALL return valid: true
 * if and only if the string matches /^[a-zA-Z0-9-]{1,30}$/.
 *
 * **Validates: Requirements 5.6, 5.7**
 */
const fc = require('fast-check');

// Extract the validate function matching the FormHandler implementation in track.html
const validPattern = /^[a-zA-Z0-9-]{1,30}$/;

function validate(trackingId) {
    if (!trackingId || trackingId.length === 0) {
        return { valid: false, message: 'Please enter a Tracking ID' };
    }
    if (!validPattern.test(trackingId)) {
        return { valid: false, message: 'Tracking ID can only contain letters, numbers, and hyphens' };
    }
    return { valid: true };
}

// Valid character set for generators
const VALID_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-';

describe('Property 4: Tracking ID validation is correct', () => {
    it('accepts all valid strings (alphanumeric + hyphens, 1-30 chars)', () => {
        fc.assert(
            fc.property(
                fc.stringOf(
                    fc.constantFrom(...VALID_CHARS),
                    { minLength: 1, maxLength: 30 }
                ),
                (validString) => {
                    const result = validate(validString);
                    return result.valid === true;
                }
            ),
            { numRuns: 100 }
        );
    });

    it('rejects empty strings', () => {
        const result = validate('');
        expect(result.valid).toBe(false);
        expect(result.message).toBe('Please enter a Tracking ID');
    });

    it('rejects strings longer than 30 characters', () => {
        fc.assert(
            fc.property(
                fc.stringOf(
                    fc.constantFrom(...VALID_CHARS),
                    { minLength: 31, maxLength: 50 }
                ),
                (longString) => {
                    const result = validate(longString);
                    return result.valid === false;
                }
            ),
            { numRuns: 100 }
        );
    });

    it('rejects strings with special characters', () => {
        // Generate strings that contain at least one character NOT in the valid set
        const specialChars = '!@#$%^&*()_+=[]{}|;:\'",.<>?/\\`~ \t\n';
        fc.assert(
            fc.property(
                fc.tuple(
                    fc.stringOf(fc.constantFrom(...VALID_CHARS), { minLength: 0, maxLength: 15 }),
                    fc.stringOf(fc.constantFrom(...specialChars), { minLength: 1, maxLength: 5 }),
                    fc.stringOf(fc.constantFrom(...VALID_CHARS), { minLength: 0, maxLength: 14 })
                ),
                ([prefix, special, suffix]) => {
                    const invalidString = prefix + special + suffix;
                    if (invalidString.length === 0) return true; // skip empty (covered separately)
                    const result = validate(invalidString);
                    return result.valid === false;
                }
            ),
            { numRuns: 100 }
        );
    });

    it('round-trip: validate(s).valid === /^[a-zA-Z0-9-]{1,30}$/.test(s) for all strings', () => {
        const referencePattern = /^[a-zA-Z0-9-]{1,30}$/;
        fc.assert(
            fc.property(
                fc.string({ minLength: 0, maxLength: 50 }),
                (s) => {
                    const result = validate(s);
                    const expected = referencePattern.test(s);
                    return result.valid === expected;
                }
            ),
            { numRuns: 100 }
        );
    });
});
