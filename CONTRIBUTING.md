# Contributing

Thank you for looking. A file format is only useful if it is boring to read and
hard to misread, so contributions are held to three rules.

1. **The bytes come first.** A change to a layout changes `SPEC.md` in the same
   pull request, and keeps every existing file readable (see SPEC §6).
2. **Measured, derived, or not claimed.** A performance number in a README,
   comment or pull request says what produced it, on what machine. If you can't
   reproduce it from this repository, say so (see `CLAIMS.md`).
3. **Tests that can fail.** A new reader or writer gets a test that would catch
   the bug it exists to prevent. The GPU suite skips itself where there is no
   adapter; anything else must run on CI.

## Setup

```
pnpm install
pnpm test
pnpm typecheck
./native/build.sh test
pnpm gpu:check        # needs a GPU
```

## Good first contributions

- Run `pnpm gpu:check` on a non-Apple GPU (Vulkan on Linux, D3D12 on Windows)
  and open an issue with the full output.
- Build `native/` on Linux or Windows and report what breaks.
- An importer: GGUF → `.aamt`, keeping the source quantisation where a kind
  matches.

## Contributor License Agreement

Before a pull request can be merged, you agree, by stating so in the pull request ("I agree to the CLA in
CONTRIBUTING.md"), that:

1. you wrote the contribution, or have the right to submit it;
2. you grant Weslyn Cory Whitehead Jr. / AsAManThinks Research a perpetual, worldwide, irrevocable, royalty-free
   licence to use, modify, sublicense and distribute your contribution under any terms, including the AAMT
   Community License and commercial licences;
3. you grant the same a licence under any patent claims you hold that your contribution necessarily infringes,
   for the same purposes;
4. you keep the copyright in your contribution.

The CLA is what lets a contribution ship in both the free and the commercial editions. This is a draft for legal
review; it may be replaced by a signed CLA before the repository is public.
