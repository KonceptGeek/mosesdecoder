#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "CoarseBiLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/Manager.h"
#include "moses/InputType.h"
#include "moses/Word.h"
#include "moses/AlignmentInfo.h"

using namespace std;

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
	ReadParameters();
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
	vector<float> newScores(m_numScoreComponents);
	newScores[0] = 1.5;
	newScores[1] = 0.3;
	newScores[2] = 0.4;
	accumulator->PlusEquals(this, newScores);

	// sparse scores
	accumulator->PlusEquals(this, "sparse-name", 2.4);

	vector<string> targetWords;
	vector<string> sourceWords;
	std::map<int, int> alignments;

	TargetPhrase currTargetPhrase = cur_hypo.GetCurrTargetPhrase();
	Manager& manager = cur_hypo.GetManager();
	const Sentence& sourceSentence =
			static_cast<const Sentence&>(manager.GetSource());

	std::set<size_t> align1 = cur_hypo.GetCurrTargetPhrase().GetAlignTerm().GetAlignmentsForTarget(cur_hypo.GetCurrTargetWordsRange().GetStartPos());
	std::set<size_t> align2 = cur_hypo.GetCurrTargetPhrase().GetAlignNonTerm().GetAlignmentsForTarget(cur_hypo.GetCurrTargetWordsRange().GetStartPos());

	std::cerr << "AlignSize1: " + boost::lexical_cast<string>(align1.size()) << std::endl;

	std::cerr << "AlignSize2: " + boost::lexical_cast<string>(align2.size()) << std::endl;

	/*
	 * Get target words. Also, get the previous hypothesised target words.
	 */
	getTargetWords(cur_hypo, targetWords, alignments);

	/*
	 * Get aligned source words in the source sentence. 
	 */
	getSourceWords(currTargetPhrase, sourceSentence, sourceWords);

	for (vector<string>::const_iterator iterator = sourceWords.begin();
			iterator != sourceWords.end(); iterator++) {
		std::cerr << " " << *iterator;
	}
	std::cerr << endl;

	for (vector<string>::const_iterator iterator = targetWords.begin();
			iterator != targetWords.end(); iterator++) {
		std::cerr << " " << *iterator;
	}
	std::cerr << endl;

	return new CoarseBiLMState(0);
}

void CoarseBiLM::getTargetWords(const Hypothesis& cur_hypo,
		std::vector<std::string> &targetWords,
		std::map<int, int> &alignments) const {

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
		targetWords.push_back(cur_hypo.GetWord(targetBegin + index).ToString());
	}

}

void CoarseBiLM::getPreviousTargetWords(const Hypothesis& cur_hypo,
		int previousWordsNeeded, std::vector<std::string> &targetWords,
		std::map<int, int> &alignments) const {
	const Hypothesis * prevHypo = cur_hypo.GetPrevHypo();
	int found = 0;

	while (prevHypo && found != previousWordsNeeded) {
		const TargetPhrase& currTargetPhrase = prevHypo->GetCurrTargetPhrase();
		for (int i = currTargetPhrase.GetSize() - 1; i > -1; i--) {
			if (found != previousWordsNeeded) {
				const Word& word = currTargetPhrase.GetWord(i);
				targetWords[found] = word.ToString();
				found++;
			} else {
				return;
			}
		}
		prevHypo = prevHypo->GetPrevHypo();
	}
}

void CoarseBiLM::getSourceWords(const TargetPhrase &targetPhrase,
		const Sentence &sourceSentence,
		std::vector<std::string> &sourceWords) const {
	/*std::size_t sourceBegin = cur_hypo.GetCurrSourceWordsRange().GetStartPos();
	 std::size_t sourceEnd = cur_hypo.GetCurrSourceWordsRange().GetEndPos();
	 if (sourceBegin != sourceEnd) {
	 for (int index = sourceBegin; index <= sourceEnd; index++) {
	 sourceWords.push_back(sourceInput.GetWord(index).ToString());
	 }
	 } else {
	 sourceWords.push_back(sourceInput.GetWord(sourceBegin).ToString());
	 }*/
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
		tgtWordToClusterId = LoadManyToOneMap(value);
	} else if (key == "srcWordToClusterId") {
		srcWordToClusterId = LoadManyToOneMap(value);
	} else if (key == "bitokenToBitokenId") {
		bitokenToBitokenId = LoadManyToOneMap(value);
	} else if (key == "bitokenIdToClusterId") {
		bitokenIdToClusterId = LoadManyToOneMap(value);
	} else if (key == "lm") {
		//TODO: load language model
	} else if (key == "ngrams") {
		nGramOrder = boost::lexical_cast<int>(value);
		//TODO: look at previous phrases
	} else {
		StatefulFeatureFunction::SetParameter(key, value);
	}
}

std::map<std::string, std::string> CoarseBiLM::LoadManyToOneMap(
		const std::string& path) {
	std::cerr << "LoadManyToOneMap Value: " + path << std::endl;

	std::map<std::string, std::string> manyToOneMap;
	manyToOneMap["testKey"] = "testValue";
	return manyToOneMap;
}

}

