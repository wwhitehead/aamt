# Security

`.aamt` readers parse untrusted files. A file that makes a reader crash, hang,
read out of bounds, or return data that does not match its identity is a
security bug.

Please report it privately to **weslyn@asamanthinks.com** with a sample file if
you can. You will get an acknowledgement within 72 hours. Please don't open a
public issue until a fix is released.

The container reader already refuses corrupt input under fuzzing (1,500 random
bit flips per run in `js/packages/container/test/tree.test.ts`). Reports that
extend that coverage to the model and substrate readers are especially welcome.
