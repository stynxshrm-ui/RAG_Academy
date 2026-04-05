### Which chunk size performed better for each query?


Chunk Size = 200
=============================================================================
**Query: What are the penalties for falls?**

Hybrid search results:
  [1] Score: 1.000 - LTIES FOR FALLS In ice dance, when skaters fall during competition, they lose points. A fall by one ...
      ✓ Contains penalty/fall info
  [2] Score: 0.465 -  lose 2 points.  INTERRUPTION PENALTIES If a performance is interrupted, penalties apply based on du...
      ✓ Contains penalty/fall info

**Query: When did ice dance become Olympic?**

Hybrid search results:
  [1] Score: 1.000 - ICE DANCE - INTRODUCTION Ice dance is a discipline of figure skating that draws from ballroom dancin...
      ✓ Contains Olympic date info
  [2] Score: 0.562 - ISTORICAL) Before 2010, the compulsory dance was the first segment. Teams performed the same pattern...


Chunk Size = 500
================================================================================
**Query: What are the penalties for falls?**

Hybrid search results:
  [1] Score: 1.000 - ICE DANCE - INTRODUCTION Ice dance is a discipline of figure skating that draws from ballroom dancin...
  [2] Score: 0.914 - LTIES FOR FALLS In ice dance, when skaters fall during competition, they lose points. A fall by one ...

**Query: When did ice dance become Olympic?**

Hybrid search results:
  [1] Score: 1.000 - ICE DANCE - INTRODUCTION Ice dance is a discipline of figure skating that draws from ballroom dancin...
      ✓ Contains Olympic date info
  [2] Score: 0.607 - ISTORICAL) Before 2010, the compulsory dance was the first segment. Teams performed the same pattern...


RECOMMENDATION
================================================================================

200 char chunks are BETTER for specific factual queries
   (Correctly retrieved penalty information)

    - Use 200 char chunks for: Specific questions, factual lookups
    - Use 500 char chunks for: Broad overview, conceptual questions