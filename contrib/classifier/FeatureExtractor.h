#ifndef moses_FeatureExtractor_h
#define moses_FeatureExtractor_h

#include "FeatureConsumer.h"
#include "ExtractorConfig.h"

#include <vector>
#include <string>
#include <exception>
#include <stdexcept>
#include <map>

namespace Classifier
{

// label index passed to the classifier, this value is not used in our setting
const int DUMMY_IDX = 1111;

// vector of words, each word is a vector of factors
typedef std::vector<std::vector<std::string> > ContextType; 

typedef std::multimap<size_t, size_t> AlignmentType;

// In DA scenario, there are multiple phrase tables. This struct
// contains scores for a phrase in one phrase-table.
struct TTableEntry
{
  std::string m_id;            // phrase-table identifier
  bool m_exists;               // does translation exist in this table 
  std::vector<float> m_scores; // translation scores (empty if m_exists == false)
};

// One translation (phrase target side). 
struct Translation
{
  vector<string> translation;              // words (surface forms) of translation
  AlignmentType m_alignment;               // phrase-internal word alignment
  std::vector<TTableEntry> m_ttableScores; // phrase scores in each phrase table
};

// extract features
class FeatureExtractor
{
public:
  FeatureExtractor(const ExtractorConfig &config, bool train);

  // Generate features for current source phrase and all its translation options, based on 
  // configuration. Calls all auxiliary Generate* methods.
  //
  // In training, reads the &losses parameter and passes them to VW. In prediction, &losses is 
  // an output variable where VW scores are written.
  void GenerateFeatures(FeatureConsumer *fc,
    const ContextType &context,
    size_t spanStart,
    size_t spanEnd,
    const std::vector<Translation> &translations,
    std::vector<float> &losses,
    std::string extraFeature = "");

private:
  const ExtractorConfig &m_config;      // Configuration of features.
  bool m_train;                         // Train or predict.

  // Get the highest probability P(e|f) associated with any of the translation options,
  // separately for each phrase table (string keys are phrase-table IDs).
  std::map<std::string, float> GetMaxProb(const std::vector<Translation> &translations);

  void GenerateContextFeatures(const ContextType &context, size_t spanStart, size_t spanEnd, FeatureConsumer *fc);
  void GeneratePhraseFactorFeatures(const ContextType &context, size_t spanStart, size_t spanEnd, FeatureConsumer *fc);
  void GenerateInternalFeatures(const std::vector<std::string> &span, FeatureConsumer *fc);
  void GenerateIndicatorFeature(const std::vector<std::string> &span, FeatureConsumer *fc);
  void GenerateConcatIndicatorFeature(const std::vector<std::string> &span1, const std::vector<std::string> &span2, FeatureConsumer *fc);
  void GenerateSTSE(const std::vector<std::string> &span1, const std::vector<std::string> &span2, const ContextType &context, size_t spanStart, size_t spanEnd, FeatureConsumer *fc);
  void GenerateBagOfWordsFeatures(const ContextType &context, size_t spanStart, size_t spanEnd, size_t factorID, FeatureConsumer *fc);
  void GeneratePairedFeatures(const std::vector<std::string> &srcPhrase,
      const std::vector<std::string> &tgtPhrase,
      const AlignmentType &align,
      FeatureConsumer *fc);
  void GenerateScoreFeatures(const std::vector<TTableEntry> &ttableScores, FeatureConsumer *fc);
  void GenerateMostFrequentFeature(const std::vector<TTableEntry> &ttableScores,
      const std::map<std::string, float> &maxProbs,
      FeatureConsumer *fc);
  void GenerateTTableEntryFeatures(const std::vector<TTableEntry> &ttableScores, FeatureConsumer *fc);
  std::string BuildContextFeature(size_t factor, int index, const std::string &value);
};

} // namespace Classifier

#endif // moses_FeatureExtractor_h