---
title: "Hugo Migration"
date: 2021-02-20T23:09:46Z
description: Hugo migration update
twitter:
  - image: https://university.tenten.co/uploads/default/original/1X/281f95ea246dfa7509bd6f9e17d63331eb8b5fdc.jpg
---

## Hugo migration
I've recently migrated my blog to Hugo instead of Jekyll. It's an ongoing process and there might
be some broken  elements (e.g. images) but will get to updating these soon. In the meantime,
enjoy the new format. Here are some things I've had to learn the hard way, and it's a bit painful
as I feel the documentation is still a little scattered, but I'm absolutely loving it so far.

### Optional: migration from Jekyll
This is for you if you have a pre-existing website built in Jekyll;

{{< highlight bash >}}
hugo import src_site dest_site
{{< /highlight >}}

This was super quick as Hugo has a built-in mechanism to import your old posts into Hugo's
format. I've found it generally worked well, though some `static` files and `assets` moved around
a bit more than I would have expected. Alas, not a dealbreaker.

## Setting up a Hugo theme
I've personally opted for this excellent looking [PaperMod theme](https://github.com/adityatelange/hugo-PaperMod).
The `exampleSite` demo was where I found things to be most helpful because I found the demo's
`config.yaml` to work out what goes where, and then fill in bits and pieces as I wanted to. This
isn't a trivial step though, and requires a bit of attention to detail to make sure the front
matter and menus look and function as expected. There's been a bit of a mix in people using
submodules (which seems to be more common?) than directly cloning the contents of the PaperMod
theme to your repo; I went for the latter as I needed to integrate some changes to enable MathJax.
I found [Geoff's post](https://geoffruddock.com/math-typesetting-in-hugo/) really helpful for this.

## Setting up search functionality
This is an example case where I thought the documentation wasn't clear from Hugo, but once enabled,
it is absolutely wicked. In your `config.yaml`, if your theme allows searching (as is the case
for PaperMod) then you would need these in your `config.yaml`:
{{< highlight yaml >}}
outputs:
  home:
  - HTML
  - JSON
{{< /highlight >}}
    
## Shortcodes, shortcodes
Shortcodes from Hugo are brilliant. It allows really nice figure integration without you having
to define separate `div` elements (I used to do this with Jekyll's minima theme...) and other
bits and pieces such as code highlight (`{{ highlight }}`) are super easy to read. I'm still
figuring this out though so far it's been straight forward.

## Test then deploy
This was the step that took me a good 2 hours, only to figure out I should have read `peaceiris`'
Github actions a little more carefully with regards to using `GITHUB_TOKEN`s. Long story short is,
if you are using Github to deploy your blog like me, then you just need to point Github to deploy
from the `gh-pages` branch via your actions (and Hugo has a minimal template for a Github action
workflow!) then you're good to go.