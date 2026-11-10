# Legacy docs domain redirect

The documentation moved to `gpjax.quantclimate.com` (Cloudflare). This
directory is the entire contents of the Netlify site that keeps the **old**
host, `docs.jaxgaussianprocesses.com`, alive as a permanent redirect.

It is not part of the docs build and is not deployed by CI. It is deployed by
hand, once, and then only again if the redirect target changes.

## Why this exists

The docs Worker can only serve hostnames in the `quantclimate.com` Cloudflare
zone, so it cannot answer for the old host. Without this, every published link to the old domain 404s — the JOSS
paper, the PyPI project page, search results, and any third-party citation.

## Deploying it

> [!WARNING]
> Deploy this to the Netlify site that serves **`docs.jaxgaussianprocesses.com`**.
> That is a different site from the one serving the `jaxgaussianprocesses.com`
> apex and `www` (the marketing site). Deploying this bundle to the marketing
> site would replace it with a redirect to the GPJax docs.
>
> Confirm before deploying — `netlify sites:list`, then check which site claims
> the `docs.` subdomain under Domain management.

That is the site named `endearing-crepe-c2d5fe`. It also hosted this repo's PR
previews (`pr-<n>--endearing-crepe-c2d5fe.netlify.app`) until CI stopped
deploying to Netlify.

```bash
npx --yes netlify-cli@latest deploy \
  --prod \
  --dir=legacy-docs-redirect \
  --site="<site-id>" \
  --message="301 legacy docs host to gpjax.quantclimate.com"
```

`--prod` is correct **here**: this is a deliberate, manual replacement of that
site's production deploy. No workflow deploys to Netlify, and none should: a CI
deploy with `--prod` would overwrite the redirect.

## Verifying

```bash
# Expect: 301, and a Location on the new host with the path preserved.
curl -sI https://docs.jaxgaussianprocesses.com/examples/regression.html \
  | grep -iE '^HTTP|^location'

# Expect: the same, through an old MkDocs-era URL. The new site's reredirect
# stub then completes the hop to /examples/classification.html.
curl -sI https://docs.jaxgaussianprocesses.com/_examples/classification/ \
  | grep -iE '^HTTP|^location'
```
