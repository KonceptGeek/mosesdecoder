#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "CoarseBiLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/Manager.h"
#include "moses/InputType.h"
#include "moses/Word.h"
#include "moses/AlignmentInfo.h"
#include "lm/model.hh"

using namespace std;
using namespace lm::ngram;

namespace Moses {
int CoarseBiLMState::Compare(const FFState& other) const {
    const CoarseBiLMState &otherState =
            static_cast<const CoarseBiLMState&>(other);

    if (m_targetLen == otherState.m_targetLen)
        return 0;
    return (m_targetLen < otherState.m_targetLen) ? -1 : +1;
}
////////////////////////////////////////////////////////////////
CoarseBiLM::CoarseBiLM(const std::string &line) :
        StatefulFeatureFunction(3, line) {
    //FactorCollection& factorFactory = FactorCollection::Instance(); //Factor Factory to use for BOS_ and EOS_
}

CoarseBiLM::~CoarseBiLM() {
    //VERBOSE(3, "Destructor Called" << endl);
    delete CoarseLM;
}

void CoarseBiLM::Load() {
    cvtSrcToClusterId = false;
    cvtBitokenToBitokenId = false;
    cvtBitokenIdToClusterId = false;
    //VERBOSE(3, "In load function, calling read parameters" << endl);
    ReadParameters();
    //VERBOSE(3, "In load function, calling read language model" << endl);
    readLanguageModel(m_lmPath.c_str());
}

void CoarseBiLM::EvaluateInIsolation(const Phrase &source,
        const TargetPhrase &targetPhrase,
        ScoreComponentCollection &scoreBreakdown,
        ScoreComponentCollection &estimatedFutureScore) const {
}

void CoarseBiLM::EvaluateWithSourceContext(const InputType &input,
        const InputPath &inputPath, const TargetPhrase &targetPhrase,
        const StackVec *stackVec, ScoreComponentCollection &scoreBreakdown,
        ScoreComponentCollection *estimatedFutureScore) const {
}

void CoarseBiLM::EvaluateTranslationOptionListWithSourceContext(
        const InputType &input,
        const TranslationOptionList &translationOptionList) const {
}

//load the sri language models
/*
 * I need to load the following many to one maps:
 * 1. english (tgt) 400 cluster ids. Map<Word, ClusterId>
 * 2. chinese (src) 400 cluster ids. Map<Word, ClusterId>
 * 3. bitoken mapping Map<TgtId-SrcId, BitokenTag>
 * 4. bitoken clusters Map<BitokenTag, ClusterId>
 *
 * I also need to load the SRILM language model. Before calling the language model to get the score I need to
 * get the current target hypothesis and the source phrase that we are currently looking at. I also need the alignments
 * I then create the bitokens using those alignments and use the many to one maps to get to bitoken cluster id phrase.
 * I then call the language model on that phrase and return the score.
 *
 */

FFState* CoarseBiLM::EvaluateWhenApplied(const Hypothesis& cur_hypo,
        const FFState* prev_state,
        ScoreComponentCollection* accumulator) const {
    // dense scores
    VERBOSE(3, "In EvaluateWhenApplied" << endl);
    vector<string> targetWords;
    vector<string> sourceWords;
    vector<string> targetWordIDs;
    vector<string> sourceWordIDs;
    vector<string> bitokens;
    vector<string> bitokenBitokenIDs;
    vector<string> bitokenWordIDs;

    float totalScore = 0.0;
    std::map<int, std::vector<int> > alignments;

    TargetPhrase currTargetPhrase = cur_hypo.GetCurrTargetPhrase();
    Manager& manager = cur_hypo.GetManager();
    //VERBOSE(3, "Fetching source sentence" << endl);
    const Sentence& sourceSentence =
            static_cast<const Sentence&>(manager.GetSource());

    //Get target words. Also, get the previous hypothesised target words.
    //VERBOSE(3, "Calling getTargetWords" << endl);
    clock_t beginTime = clock();
    getTargetWords(cur_hypo, targetWords, alignments);
    VERBOSE(3,
            "Time taken to getTargetWords: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);
    //VERBOSE(3, "Found target words: " << getStringFromList(targetWords) << endl);
    //VERBOSE(3, "replacing target words with cluster ids" << endl);
    beginTime = clock();
    replaceWordsWithClusterID(targetWords, tgtWordToClusterId, targetWordIDs);
    VERBOSE(3,
            "Time taken to replace target words with cluster ids: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);
    vector<string> wordsToScore = targetWordIDs;

    /*//VERBOSE(3, "### Printing Alignments ###" << endl);
     for(std::map<int, std::vector<int> >::const_iterator it = alignments.begin(); it != alignments.end(); it++) {
     //VERBOSE(3, it->first << ":");
     for(vector<int>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); it2++) {
     //VERBOSE(3, " " << *it2);
     }
     //VERBOSE(3, endl);
     }
     //VERBOSE(3, "DONE PRINTING ALIGNMENTS" << endl);*/

    if (cvtSrcToClusterId) {
        //Reads the source sentence and fills the sourceWords vector wit source words.
        //VERBOSE(3, "Fetching source words" << endl);
        beginTime = clock();
        getSourceWords(sourceSentence, sourceWords);
        VERBOSE(3,
                "Time taken to getSourceWords: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);

        //VERBOSE(3, "Length of Source Words: " << sourceWords.size() << endl);
        //VERBOSE(3, "Found source words: " << getStringFromList(sourceWords) << endl);
        //VERBOSE(3, "Replacing source words with cluster ids" << endl);
        beginTime = clock();
        replaceWordsWithClusterID(sourceWords, srcWordToClusterId,
                sourceWordIDs);
        VERBOSE(3,
                "Time taken to replace source words with cluster ids: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);

        //VERBOSE(3, "Replaced Words With ClusterIDs: " << getStringFromList(sourceWordIDs) << endl);
        if (cvtBitokenToBitokenId) {
            //Create bitokens.
            //VERBOSE(3, "Creating bitokens" << endl);
            beginTime = clock();
            createBitokens(sourceWordIDs, targetWordIDs, alignments, bitokens);
            VERBOSE(3,
                    "Time taken to create bitokens: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);

            //VERBOSE(3, "Found bitokens: " << getStringFromList(bitokens) << endl);
            //Replace bitokens with bitoken tags
            //VERBOSE(3, "Replacing bitokens with bitoken tags" << endl);
            beginTime = clock();
            replaceWordsWithClusterID(bitokens, bitokenToBitokenId,
                    bitokenBitokenIDs);
            VERBOSE(3,
                    "Time taken to replace bitoken with bitoken tags: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);
            //VERBOSE(3, "Replaced bitokens with bitoken tags: " << getStringFromList(bitokenBitokenIDs) << endl);
            wordsToScore = bitokenBitokenIDs;
            if (cvtBitokenIdToClusterId) {
                //Replace bitoken tags with bitoken cluster ids
                //VERBOSE(3, "Replacing bitoken tags with cluster ids" << endl);
                beginTime = clock();
                replaceWordsWithClusterID(bitokenBitokenIDs,
                        bitokenIdToClusterId, bitokenWordIDs);
                VERBOSE(3,
                        "Time taken to replace bitoken tags with cluster ids: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);
                //VERBOSE(3, "Replaced bitoken tags with cluster ids: " << getStringFromList(bitokenWordIDs) << endl);
                wordsToScore = bitokenWordIDs;
            }
        }
    }

    State state(CoarseLM->BeginSentenceState()), outState;

    //std::cerr << "Scoring Words" << std::endl;
    //VERBOSE(3, "Scoring words using language model: " << m_lmPath << endl);
    beginTime = clock();
    for (std::vector<std::string>::const_iterator iterator =
            wordsToScore.begin(); iterator != wordsToScore.end(); iterator++) {
        std::string word = *iterator;
        float score = CoarseLM->Score(state, word, outState);
        //std::cerr << "Word: " << word << ", score: " << score << std::endl;
        totalScore = totalScore + score;
        state = outState;
    }
    VERBOSE(3,
            "Time taken to calculate scores: " << float( clock () - beginTime ) / (CLOCKS_PER_SEC/1000) << endl);
    //VERBOSE(3, "Scored using language model: " << totalScore << endl);
    /*
     std::cerr << "### Printing Target Words ###" << std::endl;
     printList(targetWords);

     std::cerr << "### Printing Source Words ###" << std::endl;
     printList(sourceWords);

     std::cerr << "### Printing Target Words Cluster IDs###" << std::endl;
     printList(targetWordIDs);

     std::cerr << "### Printing Source Words Cluster IDs###" << std::endl;
     printList(sourceWordIDs);

     std::cerr << "### Printing Bitokens ###" << std::endl;
     printList(bitokens);

     std::cerr << "### Printing Bitoken-Bitoken IDs ###" << std::endl;
     printList(bitokenBitokenIDs);

     if (cvtBitokenIdToClusterId) {
     std::cerr << "#### FEATURE FUNCTION: CoarseBiLM" << std::endl;
     } else if (cvtBitokenToBitokenId) {
     std::cerr << "#### FEATURE FUNCTION: CoarseBiLMWithoutClustering" << std::endl;
     } else {
     std::cerr << "#### FEATURE FUNCTION: CoarseLM" << std::endl;
     }
     std::cerr << "### Printing wordsToScore Cluster IDs ###" << std::endl;
     printList(wordsToScore);

     std::cerr << "### Probabilties ###" << std::endl;

     std::cerr << "Total Score: " << totalScore << std::endl;
     std::cerr << "numScoreComponents: " << m_numScoreComponents << std::endl;
     std::cerr << "### Done For This Sentence ###" << std::endl;
     */
    //VERBOSE(3, "Done for this sentence" << endl);
    vector<float> newScores(m_numScoreComponents);
    newScores[0] = totalScore;
    accumulator->PlusEquals(this, newScores);

    size_t newState = getState(cur_hypo);
    return new CoarseBiLMState(newState);
}

/*
 * Get the target words from the current hypothesis and also the previous words from previous hypothesis.
 * While doing this, also get the alignments for the target words to the source words.
 *
 * targetWords and alignments are populated in this method.
 */
void CoarseBiLM::getTargetWords(const Hypothesis& cur_hypo,
        std::vector<std::string> &targetWords,
        std::map<int, vector<int> > &alignments) const {

    int currentTargetPhraseSize = cur_hypo.GetCurrTargetPhrase().GetSize();
    int previousWordsNeeded = nGramOrder - currentTargetPhraseSize;

    if (previousWordsNeeded > 0) {
        vector<string> previousWords(previousWordsNeeded);
        //Get previous target words
        getPreviousTargetWords(cur_hypo, previousWordsNeeded, previousWords,
                alignments);

        for (int i = previousWords.size() - 1; i >= 0; i--) {
            string previousWord = previousWords[i];
            boost::algorithm::trim(previousWord);
            if (!previousWord.empty()) {
                targetWords.push_back(previousWords[i]);
            }
        }
    }

    std::size_t targetBegin = cur_hypo.GetCurrTargetWordsRange().GetStartPos();
    for (int index = 0; index < currentTargetPhraseSize; index++) {
        string word = cur_hypo.GetWord(targetBegin + index).ToString();
        boost::algorithm::trim(word);
        targetWords.push_back(word);
        //find alignments for current word
        std::set<size_t> currWordAlignments =
                cur_hypo.GetCurrTargetPhrase().GetAlignTerm().GetAlignmentsForTarget(
                        index + targetBegin);
        size_t sourceBegin = cur_hypo.GetCurrSourceWordsRange().GetStartPos();

        //add alignments to map
        for (std::set<size_t>::const_iterator iterator =
                currWordAlignments.begin();
                iterator != currWordAlignments.end(); iterator++) {
            std::map<int, vector<int> >::iterator it = alignments.find(
                    index + targetBegin);
            vector<int> alignedSourceIndices;
            if (it != alignments.end()) {
                //found vector of indices
                alignedSourceIndices = it->second;
            }
            alignedSourceIndices.push_back((*iterator) + sourceBegin);
            alignments[index + targetBegin] = alignedSourceIndices;
        }
    }

}

/*
 * Get previous n (previousWordsNeeded) words from previous hypothesis. Also  get the alignments for those words.
 * targetWords and alignments are populated.
 */
void CoarseBiLM::getPreviousTargetWords(const Hypothesis& cur_hypo,
        int previousWordsNeeded, std::vector<std::string> &targetWords,
        std::map<int, vector<int> > &alignments) const {
    const Hypothesis * prevHypo = cur_hypo.GetPrevHypo();
    int found = 0;

    while (prevHypo && found != previousWordsNeeded) {
        const TargetPhrase& currTargetPhrase = prevHypo->GetCurrTargetPhrase();
        size_t tpBegin = prevHypo->GetCurrTargetWordsRange().GetStartPos();
        size_t sourceBegin = prevHypo->GetCurrSourceWordsRange().GetStartPos();

        for (int i = currTargetPhrase.GetSize() - 1; i > -1; i--) {
            if (found != previousWordsNeeded) {
                const Word& word = currTargetPhrase.GetWord(i);
                targetWords[found] = word.ToString();

                //find alignments for current word
                std::set<size_t> currWordAlignments =
                        currTargetPhrase.GetAlignTerm().GetAlignmentsForTarget(
                                i + tpBegin);
                //add alignments to map
                for (std::set<size_t>::const_iterator iterator =
                        currWordAlignments.begin();
                        iterator != currWordAlignments.end(); iterator++) {
                    std::map<int, vector<int> >::iterator it = alignments.find(
                            i + tpBegin);
                    vector<int> alignedSourceIndices;
                    if (it != alignments.end()) {
                        alignedSourceIndices = it->second;
                    }
                    alignedSourceIndices.push_back((*iterator) + sourceBegin);
                    alignments[i + tpBegin] = alignedSourceIndices;
                }

                found++;
            } else {
                return;
            }
        }
        prevHypo = prevHypo->GetPrevHypo();
    }
}

/*
 * Get the words in sourceSentence and fill the sourceWords vector.
 */
void CoarseBiLM::getSourceWords(const Sentence &sourceSentence,
        std::vector<std::string> &sourceWords) const {
    for (int index = 0; index < sourceSentence.GetSize(); index++) {
        string word = sourceSentence.GetWord(index).ToString();
        boost::algorithm::trim(word);
        sourceWords.push_back(word);
    }
}

void CoarseBiLM::replaceWordsWithClusterID(
        const std::vector<std::string> &words,
        const std::map<std::string, std::string> &clusterIdMap,
        std::vector<std::string> &wordClusterIDs) const {
//std::cerr << clusterIdMap.size() << std::endl;
    for (std::vector<std::string>::const_iterator it = words.begin();
            it != words.end(); it++) {
        std::string word = *it;
        boost::algorithm::trim(word);
        std::map<std::string, std::string>::const_iterator pos =
                clusterIdMap.find(word);
        if (pos == clusterIdMap.end()) {
            //std::cerr << "did not find a value: " << word << std::endl;
            std::map<std::string, std::string>::const_iterator unknownWord =
                    clusterIdMap.find("_UNK_"); //for embeddings get the cluster of UNK record;
            if (pos == clusterIdMap.end()) {
                //VERBOSE(3, "Inserting NULL for word: " << word << endl);
                wordClusterIDs.push_back("NULL");
            } else {
                //VERBOSE(3, "Inserting for _UNK_: " << word << endl);
                std::string clusterId = unknownWord->second;
                wordClusterIDs.push_back(clusterId);
            }
        } else {
            std::string clusterId = pos->second;
            wordClusterIDs.push_back(clusterId);
        }
    }
}

void CoarseBiLM::createBitokens(const std::vector<std::string> &sourceWords,
        const std::vector<std::string> &targetWords,
        const std::map<int, std::vector<int> > &alignments,
        std::vector<std::string> &bitokens) const {
    for (int index = 0; index < targetWords.size(); index++) {
        string targetWord = targetWords[index];
        string sourceWord = "";
        std::map<int, vector<int> >::const_iterator pos = alignments.find(
                index);
        if (pos == alignments.end()) {
            //std::cerr << "did not find a value: " << targetWord << std::endl;
            sourceWord = "NULL";
        } else {
            vector<int> sourceIndicess = pos->second;
            for (vector<int>::const_iterator it = sourceIndicess.begin();
                    it != sourceIndicess.end(); it++) {
                string tempWord = sourceWords[*it];
                //VERBOSE(3, "SourceWords Length: " << sourceWords.size() << endl);
                //VERBOSE(3, "SourceWords: " << getStringFromList(sourceWords) << endl);
                //VERBOSE(3, "Aligned TEMP WORD: " << tempWord << endl);
                sourceWord = sourceWord + "_" + tempWord;
            }
            //VERBOSE(3, "Aligned source words: " << sourceWord << endl);
            sourceWord.erase(0, 1);
            //VERBOSE(3, "Aligned source words after trimming the underscore: " << sourceWord << endl);
        }
        string bitoken = sourceWord + "-" + targetWord;
        //VERBOSE(3, "Bitoken is: " << bitoken << endl);
        //std::cerr << "Bitoken: " << bitoken << std::endl;
        bitokens.push_back(bitoken);
    }
}

size_t CoarseBiLM::getState(const Hypothesis& cur_hypo) const {

    int currentTargetPhraseSize = cur_hypo.GetCurrTargetPhrase().GetSize();
    int previousWordsNeeded = nGramOrder - currentTargetPhraseSize;
    size_t hashCode = 0;

    if (previousWordsNeeded > 0) {
        vector<string> previousWords(previousWordsNeeded);
        std::map<int, vector<int> > alignments;
        //Get previous target words
        getPreviousTargetWords(cur_hypo, previousWordsNeeded, previousWords,
                alignments);

        for (int i = previousWords.size() - 1; i >= 0; i--) {
            string previousWord = previousWords[i];
            boost::algorithm::trim(previousWord);
            if (!previousWord.empty()) {
                boost::hash_combine(hashCode, previousWords[i]);
            }
        }
    }

    std::size_t targetBegin = cur_hypo.GetCurrTargetWordsRange().GetStartPos();
    for (int index = 0; index < currentTargetPhraseSize; index++) {
        boost::hash_combine(hashCode,
                cur_hypo.GetWord(targetBegin + index).ToString());
    }

    return hashCode;
}

FFState* CoarseBiLM::EvaluateWhenApplied(const ChartHypothesis& /* cur_hypo */,
        int /* featureID - used to index the state in the previous hypotheses */,
        ScoreComponentCollection* accumulator) const {
    return new CoarseBiLMState(0);
}

void CoarseBiLM::SetParameter(const std::string& key,
        const std::string& value) {
    std::cerr << "Key: " + key << std::endl;
    if (key == "tgtWordToClusterId") {
        LoadManyToOneMap(value, tgtWordToClusterId);
    } else if (key == "srcWordToClusterId") {
        cvtSrcToClusterId = true;
        LoadManyToOneMap(value, srcWordToClusterId);
    } else if (key == "bitokenToBitokenId") {
        cvtBitokenToBitokenId = true;
        LoadManyToOneMap(value, bitokenToBitokenId);
    } else if (key == "bitokenIdToClusterId") {
        cvtBitokenIdToClusterId = true;
        LoadManyToOneMap(value, bitokenIdToClusterId);
    } else if (key == "lm") {
        m_lmPath = value;
    } else if (key == "ngrams") {
        nGramOrder = boost::lexical_cast<int>(value);
        //TODO: look at previous phrases
    } else {
        StatefulFeatureFunction::SetParameter(key, value);
    }
    std::cerr << "Parameter Set: " << key << endl;
}

void CoarseBiLM::LoadManyToOneMap(const std::string& path,
        std::map<std::string, std::string> &manyToOneMap) {
    std::cerr << "LoadManyToOneMap Value: " + path << std::endl;

    std::ifstream file;
    file.open(path.c_str(), ios::in);

    string key;
    string value;

    while (file >> key >> value) {
        boost::algorithm::trim(key);
        boost::algorithm::trim(value);
        manyToOneMap[key] = value;
    }
    file.close();
}

void CoarseBiLM::readLanguageModel(const char *lmFile) {
    CoarseLM = ConstructCoarseLM(m_lmPath);

}

void CoarseBiLM::printList(const std::vector<std::string> &listToPrint) const {
    for (std::vector<std::string>::const_iterator iterator =
            listToPrint.begin(); iterator != listToPrint.end(); iterator++) {
        std::cerr << *iterator << " ";
    }
    std::cerr << endl;
}

std::string CoarseBiLM::getStringFromList(
        const std::vector<std::string> &listToConvert) const {
    std::string result = "";
    for (std::vector<std::string>::const_iterator iterator =
            listToConvert.begin(); iterator != listToConvert.end();
            iterator++) {
        result = result + "||" + *iterator;
    }
    boost::algorithm::trim(result);
    return result;
}
}

