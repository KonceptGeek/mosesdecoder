#include <vector>
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "CoarseBiLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/Manager.h"
#include "moses/InputType.h"
#include "moses/Word.h"
#include "moses/AlignmentInfo.h"
#include "moses/Timer.h"
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
}

CoarseBiLM::~CoarseBiLM() {
    VERBOSE(3, "Destructor Called" << endl);
    delete CoarseLM100;
    delete CoarseLM1600;
    delete CoarseBiLMWithoutClustering;
    delete CoarseBiLMWithClustering;
}

void CoarseBiLM::Load() {
    VERBOSE(3, "In load function, calling read parameters" << endl);
    ReadParameters();
    VERBOSE(3, "In load function, calling read language model" << endl);
    CoarseLM100 = ConstructCoarseLM(m_lmPath100.c_str());
    CoarseLM1600 = ConstructCoarseLM(m_lmPath1600.c_str());
    CoarseBiLMWithoutClustering = ConstructCoarseLM(m_bilmPathWithoutClustering.c_str());
    CoarseBiLMWithClustering = ConstructCoarseLM(m_bilmPathWithClustering.c_str());
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

    Timer overallTimerObj;
    overallTimerObj.start("CoarseBiLM Timer");
    Timer functionTimerObj;

    vector<string> targetWords;
    vector<string> targetWordIDs100;
    vector<string> targetWordIDs1600;
    vector<string> targetWordIDs400;

    //vector<string> sourceWords;
    //vector<string> sourceWordIDs400;
    vector<string> bitokenBitokenIDs;
    vector<string> bitokenWordIDs;

    float scoreCoarseLM100 = 0.0;
    float scoreCoarseLM1600 = 0.0;
    float scoreCoarseBiLMWithoutBitokenCLustering = 0.0;
    float scoreCoarseBiLMWithBitokenCLustering = 0.0;

    std::map<int, std::vector<int> > alignments;

    Manager& manager = cur_hypo.GetManager();

    functionTimerObj.start("fetchingSourceSentence");
    const Sentence& sourceSentence = static_cast<const Sentence&>(manager.GetSource());
    functionTimerObj.stop("fetchingSourceSentence");
    VERBOSE(3, "Done fetching source sentence(" << sourceSentence.GetSize() << "): " << functionTimerObj.get_elapsed_time() << endl);

    vector<string> sourceWords(sourceSentence.GetSize(), "NULL");
    vector<string> sourceWordIDs400(sourceSentence.GetSize(), "NULL");
    VERBOSE(3, "Initialized sourceWords:("<< sourceWords.size() <<") " << getStringFromList(sourceWords) << endl);

    //Get target words. Also, get the previous hypothesised target words.
    functionTimerObj.start("getTargetWords");
    getTargetWords(cur_hypo, targetWords, targetWordIDs100, targetWordIDs1600, targetWordIDs400, alignments);
    functionTimerObj.stop("getTargetWords");
    VERBOSE(3, "TargetWords: " << getStringFromList(targetWords) << endl);
    VERBOSE(3, "TargetWords100: " << getStringFromList(targetWordIDs100) << endl);
    VERBOSE(3, "TargetWords1600: " << getStringFromList(targetWordIDs1600) << endl);
    VERBOSE(3, "TargetWords400: " << getStringFromList(targetWordIDs400) << endl);
    VERBOSE(3, "Done getTargetWords: " << functionTimerObj.get_elapsed_time() << endl);

    //get source words
    functionTimerObj.start("getSourceWords");
    getSourceWords(sourceSentence, alignments, sourceWords);
    functionTimerObj.stop("getSourceWords");
    VERBOSE(3, "SourceWords: " << getStringFromList(sourceWords) << endl);
    VERBOSE(3, "Done getSourceWords: " << functionTimerObj.get_elapsed_time() << endl);

    //create bitokens
    functionTimerObj.start("createBitokens");
    createBitokens(sourceWordIDs400, targetWordIDs400, alignments, bitokenBitokenIDs, bitokenWordIDs);
    functionTimerObj.stop("createBitokens");
    VERBOSE(3, "BitokenBitokenIDs: " << getStringFromList(bitokenBitokenIDs) << endl);
    VERBOSE(3, "BitokenWordIDs: " << getStringFromList(bitokenWordIDs) << endl);
    VERBOSE(3, "Done createBitokens: " << functionTimerObj.get_elapsed_time() << endl);

    //Score using CoarseLMs & CoarseBiLMs
    functionTimerObj.start("scoreCoarseLM100");
    scoreCoarseLM100 = getLMScore(targetWordIDs100, CoarseLM100);
    functionTimerObj.stop("scoreCoarseLM100");
    VERBOSE(3, "Done scoreCoarseLM100: " << functionTimerObj.get_elapsed_time() << endl);

    functionTimerObj.start("scoreCoarseLM1600");
    scoreCoarseLM1600 = getLMScore(targetWordIDs1600, CoarseLM1600);
    functionTimerObj.stop("scoreCoarseLM1600");
    VERBOSE(3, "Done scoreCoarseLM1600: " << functionTimerObj.get_elapsed_time() << endl);

    functionTimerObj.start("scoreCoarseBiLMWithoutBitokenCLustering");
    scoreCoarseBiLMWithoutBitokenCLustering = getLMScore(bitokenBitokenIDs, CoarseBiLMWithoutClustering);
    functionTimerObj.stop("scoreCoarseBiLMWithoutBitokenCLustering");
    VERBOSE(3, "Done scoreCoarseBiLMWithoutBitokenCLustering: " << functionTimerObj.get_elapsed_time() << endl);

    functionTimerObj.start("scoreCoarseBiLMWithBitokenCLustering");
    scoreCoarseBiLMWithBitokenCLustering = getLMScore(bitokenWordIDs, CoarseBiLMWithClustering);
    functionTimerObj.stop("scoreCoarseBiLMWithBitokenCLustering");
    VERBOSE(3, "Done scoreCoarseBiLMWithBitokenCLustering: " << functionTimerObj.get_elapsed_time() << endl);

    vector<float> newScores(m_numScoreComponents);
    newScores[0] = scoreCoarseLM100;
    newScores[1] = scoreCoarseLM1600;
    newScores[2] = scoreCoarseBiLMWithoutBitokenCLustering;
    newScores[3] = scoreCoarseBiLMWithBitokenCLustering;

    accumulator->PlusEquals(this, newScores);

    size_t newState = getState(bitokenWordIDs);
    overallTimerObj.stop("CoarseBiLM Timer");
    VERBOSE(3, "CoarseBiLM Function Took " << overallTimerObj.get_elapsed_time() << endl);

    return new CoarseBiLMState(newState);
}

float CoarseBiLM::getLMScore(const std::vector<std::string> &wordsToScore,
        const LM* languageModel) const {
    float totalScore = 0.0;
    State state(languageModel->BeginSentenceState()), outState;
    for (std::vector<std::string>::const_iterator iterator =
            wordsToScore.begin(); iterator != wordsToScore.end(); iterator++) {
        std::string word = *iterator;
        float score = languageModel->Score(state, word, outState);
        totalScore = totalScore + score;
        state = outState;
    }
    return totalScore;
}

std::string CoarseBiLM::getClusterID(
        const std::string &word,
        const std::map<std::string, std::string> &clusterIdMap) const {
    std::string result = "NULL";
    std::map<std::string, std::string>::const_iterator pos = clusterIdMap.find(word);
    if (pos == clusterIdMap.end()) {
        std::map<std::string, std::string>::const_iterator unknownWord = clusterIdMap.find("_UNK_"); //for embeddings get the cluster of UNK record;
        if (pos != clusterIdMap.end()) {
            result = unknownWord->second;
        }
    } else {
        result = pos->second;
    }
    VERBOSE(3, "GetClusterId: " << word << "-" << result << endl);
    return result;
}
/*
 * Get the target words from the current hypothesis and also the previous words from previous hypothesis.
 * While doing this, also get the alignments for the target words to the source words.
 *
 * targetWords and alignments are populated in this method.
 */
void CoarseBiLM::getTargetWords(const Hypothesis& cur_hypo,
		std::vector<std::string> &targetWords,
        std::vector<std::string> &targetWords100,
		std::vector<std::string> &targetWords1600,
		std::vector<std::string> &targetWords400,
        std::map<int, vector<int> > &alignments) const {

    int currentTargetPhraseSize = cur_hypo.GetCurrTargetPhrase().GetSize();
    int previousWordsNeeded = nGramOrder - currentTargetPhraseSize;

    if (previousWordsNeeded > 0) {
        vector<string> previousWords(previousWordsNeeded);
        //Get previous target words
        getPreviousTargetWords(cur_hypo, previousWordsNeeded, previousWords, alignments);

        for (int i = previousWords.size() - 1; i >= 0; i--) {
            string previousWord = previousWords[i];
            boost::algorithm::trim(previousWord);
            if (!previousWord.empty()) {
            	targetWords.push_back(previousWords[i]);
            	targetWords100.push_back(getClusterID(previousWords[i], tgtWordToClusterId100));
            	targetWords1600.push_back(getClusterID(previousWords[i], tgtWordToClusterId1600));
            	targetWords400.push_back(getClusterID(previousWords[i], tgtWordToClusterId400));
            }
        }
    }

    std::size_t targetBegin = cur_hypo.GetCurrTargetWordsRange().GetStartPos();
    for (int index = 0; index < currentTargetPhraseSize; index++) {
        string word = cur_hypo.GetWord(targetBegin + index).ToString();
        boost::algorithm::trim(word);
        targetWords.push_back(word);
        targetWords100.push_back(getClusterID(word, tgtWordToClusterId100));
        targetWords1600.push_back(getClusterID(word, tgtWordToClusterId1600));
        targetWords400.push_back(getClusterID(word, tgtWordToClusterId400));

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
        int previousWordsNeeded, std::vector<std::string> &previousWords,
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
                previousWords[found] = word.ToString();

                //find alignments for current word
                std::set<size_t> currWordAlignments = currTargetPhrase.GetAlignTerm().GetAlignmentsForTarget(i + tpBegin);

                //add alignments to map
                for (std::set<size_t>::const_iterator iterator = currWordAlignments.begin(); iterator != currWordAlignments.end(); iterator++) {
                    std::map<int, vector<int> >::iterator it = alignments.find(i + tpBegin);
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
void CoarseBiLM::getSourceWords(const Sentence &sourceSentence, const std::map<int, std::vector<int> > &alignments, std::vector<std::string> &sourceWords) const {
    for(std::map<int, std::vector<int> >::const_iterator it = alignments.begin(); it != alignments.end(); it++) {
        for(vector<int>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); it2++) {
            int sourceIndex = *it2;
            sourceWords[sourceIndex] = getClusterID(sourceSentence.GetWord(sourceIndex).ToString(), srcWordToClusterId400);
        }
    }
}

void CoarseBiLM::createBitokens(const std::vector<std::string> &sourceWords,
        const std::vector<std::string> &targetWords,
        const std::map<int, std::vector<int> > &alignments,
        std::vector<std::string> &bitokenBitokenIDs, std::vector<std::string> &bitokenWordIDs) const {
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
        string bitokenBitokenId = getClusterID(bitoken, bitokenToBitokenId);
        string bitokenClusterId = getClusterID(bitokenBitokenId, bitokenIdToClusterId);
        bitokenBitokenIDs.push_back(bitokenBitokenId);
        bitokenWordIDs.push_back(bitokenClusterId);
    }
}

size_t CoarseBiLM::getState(
        const std::vector<std::string> &wordsToScore) const {

    size_t hashCode = 0;

    for (int i = 0; i < wordsToScore.size(); i++) {
        boost::hash_combine(hashCode, wordsToScore[i]);
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
    if (key == "tgtWordToClusterId100") {
        LoadManyToOneMap(value, tgtWordToClusterId100);
    } else if(key == "tgtWordToClusterId1600") {
    	LoadManyToOneMap(value, tgtWordToClusterId1600);
    } else if(key == "tgtWordToClusterId400") {
    	LoadManyToOneMap(value, tgtWordToClusterId400);
    } else if (key == "srcWordToClusterId") {
        LoadManyToOneMap(value, srcWordToClusterId400);
    } else if (key == "bitokenToBitokenId") {
        LoadManyToOneMap(value, bitokenToBitokenId);
    } else if (key == "bitokenIdToClusterId") {
        LoadManyToOneMap(value, bitokenIdToClusterId);
    } else if (key == "lmCoarseLM100") {
        m_lmPath100 = value;
    } else if (key == "lmCoarseLM1600") {
        m_lmPath1600 = value;
    } else if (key == "biLMWithoutClustering") {
        m_bilmPathWithoutClustering = value;
    } else if (key == "biLMWithClustering") {
    	m_bilmPathWithClustering = value;
    } else if (key == "ngrams") {
        nGramOrder = boost::lexical_cast<int>(value);
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

