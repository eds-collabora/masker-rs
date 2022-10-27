# masker

A library for masking patterns in input data. Create a
[`Masker`](https://docs.rs/masker/latest/masker/struct.Masker.html)
with a set of input patterns - e.g. strings or slices - and your
desired mask pattern, then use it to replace those input patterns in
data you feed to it. Conceptually, this isn't much more than find and
replace with multiple patterns and a common replacement, however it
handles two awkward things:
- It allows you to stream data in chunks and still apply the
  replacements, with some small buffering as needed.
- It handles overlapping matches where replacing either one
  might leave part of the other exposed.

For the latter point, imagine our targets to mask were "captcha" and
"hat"; what happens for "capchat"? If we replace "captcha" first with
our mask, say "XXXX", we'll get "XXXXt" as our output. If we replace
"hat" first, we'll get "captcXXXX". This library will produce "XXXX"
only - no part of any masked string that appears in the text will be
revealed, even when it overlaps with other masked strings.

Example:

```rust
use masker::Masker;

let m = Masker::new(&["frog", "cat"], "XXXX");
let mut cm = m.mask_chunks();
let mut v = Vec::new();
v.extend(cm.mask_chunk("the ba"));
v.extend(cm.mask_chunk("d f"));
v.extend(cm.mask_chunk("rog sat on the c"));
v.extend(cm.mask_chunk("at"));
v.extend(cm.finish());
```

## License

This code is made available under either an
[Apache-2.0](https://opensource.org/licenses/Apache-2.0) or an [MIT
license](https://opensource.org/licenses/MIT).
