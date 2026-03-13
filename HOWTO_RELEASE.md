How to issue a salem release.

Prerequisites
-------------

You must set up the corresponding publisher on PyPI and TestPyPI.
The settings are:
Project name: salem
Owner: fmaussion
Repository name: salem
Workflow name: create-release.yml
Environment name: pypi-release

In Github, under Settings -> Environments, create an environment named
"pypi-release", and click "configure environment". Add a required reviewer.
These should be restricted to highly-trusted contributors to salem, as it
grants the power to publish on PyPI.

*IMPORTANT*
The workflow and environment names on PyPI and GitHub should be identical,
including the file extension (yml, not yaml).

Release steps
-------------

1. Go to your local main repository (not the fork) and ensure your master branch
   is synced.
      ```
      git fetch --all
      git checkout master
      git pull upstream master
      ```
2. Check whats-new.rst and the docs.
   - Make sure "What's New" is complete (check the date!)
   - Add a brief summary note describing the release at the top.
   - Check index.rst and update the version.
3. If you have any doubts, run the full test suite one final time!
      ``pytest --run-slow --mpl .``
4. On the master branch, commit the release in git.
      ```
      git commit -a -m 'Release v0.X.Y'
      ```
5. Tag the release and push to Github.
      ```
      git tag -a v1.X.Y -m 'v1.X.Y'
      git push origin tag v1.X.Y
      ```
7. Go to https://github.com/fmaussion/salem/actions and approve the release.
   **This is the final step, you cannot take this back!** The release will be
   built and published to PyPI and GitHub Releases automatically. You can
   monitor the progress in the Actions tab of the repository.
8. Update the stable branch (used by ReadTheDocs) and switch back to master:
      ```
      git checkout stable
      git rebase master
      git push origin stable
      git checkout master
      ```
      It's OK to force push to 'stable' if necessary. We also update the stable
      branch with `git cherrypick` for documentation-only fixes that apply to
      the current released version.
9. Add a section for the next release (`v.X.(Y+1)`) to `doc/whats-new.rst`.
10. Commit your changes and push to master again:
      ```
      git commit -a -m 'Revert to dev version'
      git push origin master
      ```
11. Issue the release on GitHub. Click on "Draft a new release" at
    https://github.com/fmaussion/salem/releases. Type in the version number, but
    don't bother to describe it -- we maintain that on the docs instead.
12. Update the docs. Login to https://readthedocs.org/projects/salem/versions/
    and switch your new release tag (at the bottom) from "Inactive" to "Active".
    It should now build automatically.
13. Issue the release announcement!
