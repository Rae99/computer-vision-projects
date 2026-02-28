# Assessment #1: Features — Entropy

**CS5330**
**2026 Spring**
**assessment 1**
**Junrui Ding**

---

## Reflection

I thought I had a good understanding of entropy before this assignment but I still learnt something new from it.

Mostly important, Working on this topic corrected a few misconceptions I had going in.

Before running the code, I assumed the grass region in my photo would have high entropy — it has visible light and dark stripes from shadows, so I expected a lot of pixel value variation. It turns out not the case, the entropy came out moderate (6.61), lower than the treeline region. After looking at the histogram, I realized why: the transition between light and dark in the shadow stripes is sharp, not gradual. The pixels that actually fall on the transition edges are relatively few, so they don't contribute much to the overall distribution. The majority of grass pixels cluster in a narrow dark-green range, making the histogram concentrated rather than spread out, and a concentrated histogram means lower entropy, not higher.

So now I know how to correclty read and analyze an image in terms of its entropy.
Entropy measures the spread of the distribution, not visual complexity in the intuitive sense. A region can look interesting to the eye while having moderate entropy if its pixel values are still clustered.

---

## Acknowledgements

I used Claude (Anthropic) as a learning aid throughout this project. I asked it to explain concepts like why log is used in the Shannon formula, and to help me write and debug C++ OpenCV code for computing entropy and generating visualizations. All slides and the final video are my own work.

- **Why log is used in the Shannon formula.** I understood that $p_i$ represented probability, but wasn't sure why taking the log was the right way to measure "surprise." Claude explained that $-\log_2 p_i$ has a natural interpretation: it measures how many yes/no questions you'd need to identify the outcome. A rare event (small $p$) requires more questions and produces a larger value, while a certain event ($p = 1$) needs zero questions and gives zero. This additivity property — that the surprise of two independent events equals the sum of their individual surprises — is also why log is the unique function that satisfies Shannon's definition of entropy.
