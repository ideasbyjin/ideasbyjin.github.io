---
categories:
- bioinf
date: "2020-05-16T00:00:00Z"
image: https://ideasbyjin.github.io/assets/antibody-structure.png
mathjax: true
summary: A primer on antibodies
title: What are antibodies, anyway?
---

The COVID-19 Coronavirus has had a once-in-a-generation type of impact on our lives. Its devasatation has reached far and
wide; whether it's our jobs, our day-to-day lives, or even loved ones. The virus remains an enigma - some individuals seem
to suffer more than others, and the virus' behaviour seems to be more and more unpredictable as the days go by. One strategy
that has been hailed as an effective method to manage the crisis is _testing_: finding those who have been infected with
the virus. Here in the UK, the government has recently turned to getting [tests from Roche](https://www.bbc.co.uk/news/health-52656808).

These diagnostic tests from Roche are _serology tests_: they take a sample of an individual's blood, and test for _antibodies_. 
Hold on a minute!
 
* What exactly are antibodies? 
* What does having antibodies mean? 
* How does this help with testing for COVID-19? 

In this post, I'll do a little primer on what antibodies are, and hopefully show you why they are so cool.

If you have...
* **30 seconds**: They are Y-shaped proteins that can recognise foreign molecules. Good antibodies lead to the rest of
your immune system helping to clear out that foreign molecule.
* **10 minutes**: Go on.
> NB: Unlike my other posts this one will be fairly biology heavy, you've been warned. I'll have a jargon buster session at the end.

### The antibody molecule

Antibodies are proteins. In humans, they look like the letter Y, as shown below:

<div style="text-align: center">
    <img src="/assets/antibody-structure.png" width = "50%">
</div>

The antibody molecule has two pairs of two protein chains:

* Two "heavy" chains (green)
* Two "light" chains (cyan)

Each chain can be sub-divided into globular units, or _domains_. A single chain can have:

* A variable domain (VH, VL)
* A constant domain (CH, CL)

The two variable domains from each chain (VH+VL) form the **Fv region**. Within the Fv are six loops, known as the CDR
loops.

This is key: the amino acid sequence (and thus the shape) of the CDR loops determines whether an antibody can bind
things, e.g. the COVID-19 virus, with sufficient strength. Put another way, I like to think of the entire Fv region as
being similar to our hands, and the CDR loops are like our fingers. Depending (largely) on the shape of your fingers,
you can hold on to different sized objects. 

For example, this is the molecular structure of an antibody binding its target molecule, also known as the _antigen_:

<div style="text-align: center">
    <img src="/assets/antibody-binding.png" width = "50%">
</div>

Notice that the pink blob (antigen) is in close contact with the CDR loops (coloured red, yellow, purple...). Furthermore,
as the antibody has two Fv regions, it can bind to two antigens simultaneously. One more thing to mention here is that
the combined set of CDR loops recognise a particular spot on the antigen, known as the _epitope_. This means that, in
theory, that pink blob can be recognised by lots of different antibodies throughout its entire surface.

### The antibody's origins

Antibodies are synthesised from a type of white blood cell called the B-lymphocyte. To cut a couple weeks' and textbook chapters
short, every person has billions of B-cells, each of which produces one antibody. Each B-cell performs a process known
as V(D)J recombination to randomly stitch together parts of the genome to produce an antibody: 

<div style="text-align: center">
    <img src="/assets/antibody-vdj.png" width = "50%">
</div>

To start with, an antibody starts its life as a protein on the surface of the B-cell (technically called a B-cell receptor). 
When a circulating B-cell comes in contact with its target antigen, the B-cell converts the B-cell receptor to a soluble
form that circulates in the bloodstream. This entire process takes a few weeks after the initial infection.

Altogether, if COVID-19 binding antibodies are detected in an individual, one can assume that they have been infected with
the virus a couple of weeks prior to the time of taking a test.  

### What do antibodies do again?

Basically antibodies bind things. What's important is what happens afterward. Antibodies trigger a series of responses
from the immune system after recognising the antigen.

<div style="text-align: center">
    <img src="/assets/antibody-response.png" width = "50%">
</div>

Broadly speaking, in all three cases, the antibody acts as a "flag" that tells other parts of the immune response to
do something about the antigen. Against viruses, antibodies can also stop them from exiting an infected human cell. In
fact, the range of "healthy" responses mediated by antibodies is discussed in this paper by [Tay et al](https://www.nature.com/articles/s41577-020-0311-8/).

<figure>
    <div style="text-align: center">
        <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41564-019-0392-y/MediaObjects/41564_2019_392_Fig1_HTML.png?as=webp" width = "50%">
        <figcaption>Figure from <a href="https://www.nature.com/articles/s41564-019-0392-y">Murin et al.</a> The left hand
        side of the figure shows the broad life cycle of a virus, and the right-hand side shows how antibodies can block
        viruses from further entry or exit.</figcaption>
    </div>
</figure>

Are all antibodies equal? Not quite. In some unfortunate cases, antibodies might facilitate viral entry, as shown in panel B below:  

<figure>
    <div style="text-align: center">
        <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41577-020-0321-6/MediaObjects/41577_2020_321_Fig1_HTML.png?as=webp" width = "30%">
        <figcaption>Figure from <a href="https://www.nature.com/articles/s41577-020-0321-6">Iwasaki and Yang.</a></figcaption>
    </div>
</figure>

This phenomenon, known as antibody-directed enhancement, has been documented in that paper by Tay et al. that I described
above. Other reviews like this one by [de Alwis et al](https://www.sciencedirect.com/science/article/pii/S2352396420301432)
speculate that antibodies can cause hyper-active immune responses, which can also be detrimental.

### Then how do these antibody tests work?

Let's recap everything discussed above:

* Antibodies have two pairs of heavy-light chain pairs, leading to two identical binding sites (Fv regions)
* An antibody comes from a B-cell after infection, and is then released to the bloodstream
* Antibodies can trigger immune responses in individuals, but sometimes those antibodies may not be so useful

Diagnostic tests like those from Roche leverage the first two points. It looks to see if a patient has developed antibodies
against COVID-19 (whether it's protective or not). This is done by a "double antigen sandwich":
 
<figure>
    <div style="text-align: center">
        <img src="/assets/antibody-doublesandwich.png" width = "25%">
        <img src="/assets/antibody-detection.png" width = "25%">
        <figcaption>Figures from <a href="https://www.roche.com/about/business/diagnostics/medical_value/products_and_solutions/antibody-testing-covid-19.htm">Roche.</a></figcaption>
    </div>
</figure>

Essentially, the test counts on the fact that a single antibody molecule can bind two antigens - one on each Fv. Each
antigen is either:
* Hooked up to a biotin molecule (which acts as an anchor to hold antibodies in place), or
* It is ruthenylated, making the antigen "light up" in the testing kit

I'm deliberately skipping a lot of the details of the test, but these are the basic biological details. While the test
provides valuable information for whether or not you have antibodies against COVID-19, the test does not confirm whether
you are immune to the virus. To confirm this, other tests would be necessary. For example, antibodies would have to be
isolated and tested to see if they can neutralise the virus in a separate lab experiment.

### Wrap-up

I think what is extremely remarkable about antibodies is that they are natural, in-built defence mechanisms. Almost
magically, they can protect us from viruses that we have never seen before, like COVID-19. By understanding their biological
structure and mechanism, this has allowed us to develop instrumental tools like Roche's diagnostic tests.

This post was intended to be a basic primer that (hopefully) landed somewhere between a Guardian article and a journal
article (so, a textbook I guess?) There is still so much more we can talk about in this space, whether it's:

* How do we design external antibodies to fight the virus?
* What makes antibodies go bad?
* How can artificial intelligence approaches alongside antibodies help to fight COVID-19?

But perhaps that's for another time.

### Acknowledgements / References

For any _uncited_ figures, they are almost all from my PhD thesis. The exception is the picture of an antibody binding
two pink blobs, which I made for a presentation.

Otherwise I've referred to the figures directly and I'm incredibly grateful for scientific illustrators and creators at
NPG and Roche. Good illustrations go a long way to helping everyone understand how things work!