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

    if (hashCode == otherState.hashCode)
        return 0;
    return (hashCode < otherState.hashCode) ? -1 : +1;
}
////////////////////////////////////////////////////////////////
CoarseBiLM::CoarseBiLM(const std::string &line) :
        StatefulFeatureFunction(3, line) {
}

CoarseBiLM::~CoarseBiLM() {
    VERBOSE(3, "Destructor Called" << endl);
    delete CoarseLM100;
    delete CoarseLM1600;
    delete CoarseBiLMWithClustering;
}

void CoarseBiLM::Load() {
    VERBOSE(3, "In load function, calling read parameters" << endl);
    ReadParameters();
    VERBOSE(3, "In load function, calling read language model" << endl);
    CoarseLM100 = ConstructCoarseLM(m_lmPath100.c_str());
    CoarseLM1600 = ConstructCoarseLM(m_lmPath1600.c_str());
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

const FFState* CoarseBiLM::EmptyHypothesisState(const InputType &input) const
{
  VERBOSE(3,"CoarseBiLM::EmptyHypothesisState()" << endl);

  State lm100StartingState = CoarseLM100->BeginSentenceState();
  State lm1600StartingState = CoarseLM1600->BeginSentenceState();
  State lmBitokenWithClusteringState = CoarseBiLMWithClustering->BeginSentenceState();
  vector<string> sourceWords;


  return new CoarseBiLMState(0, sourceWords, lm100StartingState, lm1600StartingState, lmBitokenWithClusteringState);
}

//load the sri language models
/*
 * I need to load the following many to one maps:
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
    const CoarseBiLMState * prevCoarseBiLMState = NULL;
    if(cur_hypo.GetWordsBitmap().GetNumWordsCovered() == 0) {
    	prev_state = NULL;
    	VERBOSE(3, "reseting state" << endl);
    } else {
    	VERBOSE(3, "wordsRange: " << cur_hypo.GetCurrSourceWordsRange() << endl);
        VERBOSE(3, "cannot reset state: " << cur_hypo.GetWordsBitmap().GetNumWordsCovered() << endl);
    }
    if(prev_state != NULL) {
        prevCoarseBiLMState =  static_cast<const CoarseBiLMState *>(prev_state);
    } else {
        VERBOSE(3, "state is null" << endl);
    }

    Timer overallTimerObj;
    Timer functionTimerObj;
    overallTimerObj.start("CoarseBiLM Timer");
    vector<string> sourceWords;
    vector<string> targetWords;
    vector<string> targetWords100;
    vector<string> targetWords1600;
    vector<string> targetWords400;
    vector<string> bitokenWordIDs;

    float scoreCoarseLM100 = 0.0;
    float scoreCoarseLM1600 = 0.0;
    float scoreCoarseBiLMWithBitokenCLustering = 0.0;

    //functionTimerObj.start("getSourceWords");
    if(prev_state != NULL && prevCoarseBiLMState->getSourceWords().size() > 0) {
        sourceWords = prevCoarseBiLMState->getSourceWords();
    } else {
        //functionTimerObj.start("fetchingSourceSentence");
        const Sentence& sourceSentence = static_cast<const Sentence&>(cur_hypo.GetManager().GetSource());
        //functionTimerObj.stop("fetchingSourceSentence");
        VERBOSE(3, "Done fetching source sentence: " << functionTimerObj.get_elapsed_time() << endl);

        getSourceWords(sourceSentence, sourceWords);
        //VERBOSE(3, "resetting state cos of source words");
        //prev_state = NULL;

    }
    //functionTimerObj.stop("getSourceWords");
    VERBOSE(3, "Done getting source words: " << getStringFromList(sourceWords) << ". It took: " << functionTimerObj.get_elapsed_time() << endl);

    State lm100StartingState;
    if(prev_state != NULL) {
        lm100StartingState = prevCoarseBiLMState->getLm100State();
    } else {
        lm100StartingState = State(CoarseLM100->BeginSentenceState());
    }

    State lm1600StartingState;
    if(prev_state != NULL) {
        lm1600StartingState = prevCoarseBiLMState->getLm1600State();
    } else {
        lm1600StartingState = State(CoarseLM1600->BeginSentenceState());
    }

    State lmBitokenWithClusteringState;
    if(prev_state != NULL) {
        lmBitokenWithClusteringState = prevCoarseBiLMState->getBiLmWithClusteringState();
    } else {
        lmBitokenWithClusteringState = State(CoarseBiLMWithClustering->BeginSentenceState());
    }


    const TargetPhrase& currTargetPhrase = cur_hypo.GetCurrTargetPhrase();
    int currentTargetPhraseSize = currTargetPhrase.GetSize();
    size_t sourceBegin = cur_hypo.GetCurrSourceWordsRange().GetStartPos();
    functionTimerObj.start("scoring target words");
    for (int index = 0; index < currentTargetPhraseSize; index++) {
        string word = currTargetPhrase.GetWord(index).ToString();
        boost::algorithm::trim(word);
        string targetWord100 = getClusterID(word, tgtWordToClusterId100);
        string targetWord1600 = getClusterID(word, tgtWordToClusterId1600);
        string bitokenTargetWord = word;

        targetWords.push_back(word);
        targetWords100.push_back(targetWord100);
        targetWords1600.push_back(targetWord1600);
        targetWords400.push_back(bitokenTargetWord);

        //find alignments for current word
        std::set<size_t> currWordAlignments = currTargetPhrase.GetAlignTerm().GetAlignmentsForTarget(index);
        string bitokenSourceWord = "";
        if(currWordAlignments.size() > 0) {
        	VERBOSE(3, "gotCurrWordAlignments" << endl);
            for (std::set<size_t>::const_iterator iterator = currWordAlignments.begin(); iterator != currWordAlignments.end(); iterator++) {
                string tempWord = sourceWords[*iterator+sourceBegin];
                //VERBOSE(3, "gotTempWord: " << tempWord << endl);
                bitokenSourceWord = bitokenSourceWord + "_" + tempWord;
                //VERBOSE(3, "bitokenSourceWord: " << bitokenSourceWord << endl);
            }
            //VERBOSE(3, "done iterating currWordAlignments" << endl);
            bitokenSourceWord.erase(0, 1);
        } else {
            bitokenSourceWord = "NULL";
        }
        VERBOSE(3, "bitokenSourceWord: " << bitokenSourceWord << endl);
        string bitoken = bitokenSourceWord + "-" + bitokenTargetWord;
        VERBOSE(3, "bitoken: " << bitoken << endl);
        string bitokenBitokenId = getClusterID(bitoken, bitokenToBitokenId);
        VERBOSE(3, "bitokenBitokenId: " << bitokenBitokenId << endl);
        string bitokenClusterId = getClusterID(bitokenBitokenId, bitokenIdToClusterId);
        VERBOSE(3, "bitokenClusterId: " << bitokenClusterId << endl);

        bitokenWordIDs.push_back(bitokenClusterId);

        scoreCoarseLM100 += getLMScore(targetWord100, CoarseLM100, lm100StartingState);
        scoreCoarseLM1600 += getLMScore(targetWord1600, CoarseLM1600, lm1600StartingState);
        scoreCoarseBiLMWithBitokenCLustering += getLMScore(bitokenClusterId, CoarseBiLMWithClustering, lmBitokenWithClusteringState);
    }

    functionTimerObj.stop("scoring target words");
    VERBOSE(3, "Done scoring target words. It took: " << functionTimerObj.get_elapsed_time() << endl);


    VERBOSE(3, "Done creating bitokens: " << functionTimerObj.get_elapsed_time() << endl);
    VERBOSE(3, "TargetWords: " << getStringFromList(targetWords) << endl);
    VERBOSE(3, "TargetWords400: " << getStringFromList(targetWords400) << endl);
    VERBOSE(3, "TargetWords100: " << getStringFromList(targetWords100) << endl);
    VERBOSE(3, "TargetWords1600: " << getStringFromList(targetWords1600) << endl);
    VERBOSE(3, "BitokenWordIDs: " << getStringFromList(bitokenWordIDs) << endl);

    VERBOSE(3, "scoreCoarseLM100: " << scoreCoarseLM100 << endl);
    VERBOSE(3, "scoreCoarseLM1600: " << scoreCoarseLM1600 << endl);
    VERBOSE(3, "scoreCoarseBiLMWithBitokenCLustering: " << scoreCoarseBiLMWithBitokenCLustering << endl);
    
    /*if(scoreCoarseLM100 < -400.0) {
        scoreCoarseLM100 = -400.0;
    } else if(scoreCoarseLM100 > 0.0) {
        scoreCoarseLM100 = -1.0;
    }
    if(scoreCoarseLM1600 < -400.0) {
        scoreCoarseLM1600 = -400.0;
    } else if(scoreCoarseLM1600 > 0.0) {
        scoreCoarseLM1600 = -1.0;
    }
    if(scoreCoarseBiLMWithoutBitokenCLustering < -400.0) {
        scoreCoarseBiLMWithoutBitokenCLustering = -400.0;
    } else if(scoreCoarseBiLMWithoutBitokenCLustering > 0.0) {
        scoreCoarseBiLMWithoutBitokenCLustering = -1.0;
    }
    if(scoreCoarseBiLMWithBitokenCLustering < -400.0) {
        scoreCoarseBiLMWithBitokenCLustering = -400.0;
    } else if(scoreCoarseBiLMWithBitokenCLustering > 0.0) {
        scoreCoarseBiLMWithBitokenCLustering = -1.0;
    }*/
    VERBOSE(3, "ScoreComponents " << m_numScoreComponents << endl);
    vector<float> newScores(m_numScoreComponents);
    newScores[0] = scoreCoarseLM100;
    newScores[1] = scoreCoarseLM1600;
    newScores[2] = scoreCoarseBiLMWithBitokenCLustering;

    accumulator->PlusEquals(this, newScores);

    overallTimerObj.start("CoarseBiLM Timer");
    VERBOSE(3, "DONE CoarseBiLM: " << overallTimerObj.get_elapsed_time() << endl);

    size_t newState = getState(bitokenWordIDs);
    return new CoarseBiLMState(newState, sourceWords, lm100StartingState, lm1600StartingState, lmBitokenWithClusteringState);
}

float CoarseBiLM::getLMScore(const std::string &wordToScore, const LM* languageModel, lm::ngram::State &state) const {
    State outState;
    float score = languageModel->Score(state, wordToScore, outState);
    state = outState;
    return score;
}

std::string CoarseBiLM::getClusterID(
        const std::string &word,
        const boost::unordered_map<std::string, std::string> &clusterIdMap) const {
    std::string result = "NULL";
    boost::unordered_map<std::string, std::string>::const_iterator pos = clusterIdMap.find(word);
    if (pos == clusterIdMap.end()) {
        /*boost::unordered_map<std::string, std::string>::const_iterator unknownWord = clusterIdMap.find("_UNK_"); //for embeddings get the cluster of UNK record;
        if (pos != clusterIdMap.end()) {
            result = unknownWord->second;
        }*/
    } else {
        result = pos->second;
    }
    //VERBOSE(3, "GetClusterId: " << word << "-" << result << endl);
    return result;
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
	UTIL_THROW2("Chart decoding not support by UTIL_THROW2");
}

void CoarseBiLM::SetParameter(const std::string& key,
        const std::string& value) {
    std::cerr << "Key: " + key << std::endl;
    if (key == "tgtWordToClusterId100") {
        LoadManyToOneMap(value, tgtWordToClusterId100);
    } else if(key == "tgtWordToClusterId1600") {
        LoadManyToOneMap(value, tgtWordToClusterId1600);
    }  else if (key == "bitokenToBitokenId") {
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
        boost::unordered_map<std::string, std::string> &manyToOneMap) {
    std::cerr << "LoadManyToOneMap Value: " + path << std::endl;

    std::ifstream file;
    file.open(path.c_str(), ios::in);

    string key;
    string value;

    while (file >> key >> value) {
        boost::algorithm::trim(key);
        boost::algorithm::trim(value);
        manyToOneMap[key] = value;
        //VERBOSE(3, key << "-" << value << endl);
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

std::string CoarseBiLM::getStringFromIntList(const lm::WordIndex words[7]) const {
    ostringstream result;
    for(int i = 0; i < 8; i++) {
        result << words[i];
        result << "||";
    }
    return result.str();
}
}

