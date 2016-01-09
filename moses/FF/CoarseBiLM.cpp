#include <vector>
#include "CoarseBiLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"

using namespace std;

namespace Moses
{
int CoarseBiLMState::Compare(const FFState& other) const
{
  const CoarseBiLMState &otherState = static_cast<const CoarseBiLMState&>(other);

  if (m_targetLen == otherState.m_targetLen)
    return 0;
  return (m_targetLen < otherState.m_targetLen) ? -1 : +1;
}

////////////////////////////////////////////////////////////////
CoarseBiLM::CoarseBiLM(const std::string &line)
  :StatefulFeatureFunction(3, line)
{
  ReadParameters();
}

void CoarseBiLM::EvaluateInIsolation(const Phrase &source
    , const TargetPhrase &targetPhrase
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection &estimatedFutureScore) const
{}

void CoarseBiLM::EvaluateWithSourceContext(const InputType &input
    , const InputPath &inputPath
    , const TargetPhrase &targetPhrase
    , const StackVec *stackVec
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection *estimatedFutureScore) const
{}

void CoarseBiLM::EvaluateTranslationOptionListWithSourceContext(const InputType &input
    , const TranslationOptionList &translationOptionList) const
{}

FFState* CoarseBiLM::EvaluateWhenApplied(
  const Hypothesis& cur_hypo,
  const FFState* prev_state,
  ScoreComponentCollection* accumulator) const
{
  // dense scores
  vector<float> newScores(m_numScoreComponents);
  newScores[0] = 1.5;
  newScores[1] = 0.3;
  newScores[2] = 0.4;
  accumulator->PlusEquals(this, newScores);

  // sparse scores
  accumulator->PlusEquals(this, "sparse-name", 2.4);

  TargetPhrase currTargetPhrase = cur_hypo.GetCurrTargetPhrase();

  string tgt = cur_hypo.GetTargetPhraseStringRep();
  AlignmentInfo alignTerm = currTargetPhrase.GetAlignTerm();
  AlignmentInfo::CollType alignments = alignTerm.GetAlignments();
  string sourcePhraseStringRep = cur_hypo.GetSourcePhraseStringRep();

  // int targetLen = cur_hypo.GetCurrTargetPhrase().GetSize(); // ??? [UG]

  //need to load 2 language models LM-400bi(400En, 400Cn)
  //need to read the many to one mapping files for 400En, 400Cn and 400bi, containing the word:clusterId
  //get the 8 target words and replace with clusterId
  //get the 8 source words and replace with cluster id
  //get the alignments and create bitokens
  //replace bitokens with tags
  //get clusterIds for bitoken tags
  //query language model to get probability

  const std::size_t targetBegin = cur_hypo.GetCurrTargetWordsRange().GetStartPos();
  //[begin, end) in STL-like fashion.
  const std::size_t targetEnd = cur_hypo.GetCurrTargetWordsRange().GetEndPos();
  //const std::size_t adjust_end = std::min(end, begin + m_ngram->Order() - 1);

  const std::size_t sourceBegin = cur_hypo.GetCurrSourceWordsRange().GetStartPos();
  const std::size_t sourceEnd = cur_hypo.GetCurrSourceWordsRange().GetEndPos();

  std::cerr << targetBegin << std::endl;
  std::cerr << targetEnd << std::endl;
  std::cerr << sourceBegin << std::endl;
  std::cerr << sourceEnd << std::endl;

  std::cerr << cur_hypo.GetWord(targetBegin).ToString() << std::endl;
  std::cerr << cur_hypo.GetWord(targetEnd).ToString() << std::endl;

  std::cerr << cur_hypo.GetCurrWord(sourceBegin).ToString() << std::endl;

  Manager& manager = cur_hypo.GetManager();
  const Sentence& source_sent = static_cast<const Sentence&>(manager.GetSource());

  std::cerr << source_sent.GetWord(sourceBegin).ToString() << std::endl;
  std::cerr << source_sent.GetWord(sourceEnd).ToString() << std::endl;

  return new CoarseBiLMState(0);
}

FFState* CoarseBiLM::EvaluateWhenApplied(
  const ChartHypothesis& /* cur_hypo */,
  int /* featureID - used to index the state in the previous hypotheses */,
  ScoreComponentCollection* accumulator) const
{
  return new CoarseBiLMState(0);
}

void CoarseBiLM::SetParameter(const std::string& key, const std::string& value)
{
  if (key == "arg") {
    // set value here
  } else {
    StatefulFeatureFunction::SetParameter(key, value);
  }
}

}

