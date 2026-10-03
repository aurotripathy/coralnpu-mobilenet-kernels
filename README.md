# Archived — moved to `mobile-yam-gemma`

This repo (patches + overlay on a pinned upstream coralnpu commit) has been
superseded by a standalone full-tree repo:

**https://github.com/aurotripathy/mobile-yam-gemma**

That repo is a copy of [`google-coral/coralnpu`](https://github.com/google-coral/coralnpu)
at `c9d3cd88` (this repo's `BASE_COMMIT`) with the same work committed on top:
optimized MobileNet int8 conv kernels, the npusim MobileNet verification flow,
the Gemma 3 270M bare-metal decoder, the YAMNet example, and the RTL MobileNet
run / vl=0 deadlock repro. Reproduce with:

```bash
git clone https://github.com/aurotripathy/mobile-yam-gemma.git
cd mobile-yam-gemma
bazel run //tests/npusim_examples/mobilenet:npusim_verify_val10
```

See its [`CHANGES.md`](https://github.com/aurotripathy/mobile-yam-gemma/blob/main/CHANGES.md)
for the full description. The exact delta over upstream is
`git log c9d3cd88..HEAD` there.

The `patches/`, `overlay/`, and `apply.sh` in this repo still work against
`BASE_COMMIT` but will not receive further updates. The last content commit
here is `9ce3ea7`, which corresponds to `3250f1cc` in the new repo.
