package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	_ "image/jpeg"

	"github.com/sirupsen/logrus"
	"github.com/tmc/langchaingo/llms"
)

const (
	ownBusinessName = "aleksander goli usługi it"
	ownBusinessNIP  = "6351860955"
)

// getSuggestedCorrespondent generates a suggested correspondent for a document using the LLM
func (app *App) getSuggestedCorrespondent(ctx context.Context, content string, suggestedTitle string, availableCorrespondents []string, correspondentBlackList []string) (string, error) {
	likelyLanguage := getLikelyLanguage()

	templateMutex.RLock()
	defer templateMutex.RUnlock()

	// Get available tokens for content
	templateData := map[string]interface{}{
		"Language":                likelyLanguage,
		"AvailableCorrespondents": availableCorrespondents,
		"BlackList":               correspondentBlackList,
		"Title":                   suggestedTitle,
	}

	availableTokens, err := getAvailableTokensForContent(correspondentTemplate, templateData)
	if err != nil {
		return "", fmt.Errorf("error calculating available tokens: %v", err)
	}

	// Truncate content if needed
	truncatedContent, err := truncateContentByTokens(content, availableTokens)
	if err != nil {
		return "", fmt.Errorf("error truncating content: %v", err)
	}

	// Execute template with truncated content
	var promptBuffer bytes.Buffer
	templateData["Content"] = truncatedContent
	err = correspondentTemplate.Execute(&promptBuffer, templateData)
	if err != nil {
		return "", fmt.Errorf("error executing correspondent template: %v", err)
	}

	prompt := promptBuffer.String()
	log.Debugf("Correspondent suggestion prompt: %s", prompt)

	completion, err := app.LLM.GenerateContent(ctx, []llms.MessageContent{
		{
			Parts: []llms.ContentPart{
				llms.TextContent{
					Text: prompt,
				},
			},
			Role: llms.ChatMessageTypeHuman,
		},
	}, llms.WithTemperature(0))
	if err != nil {
		return "", fmt.Errorf("error getting response from LLM: %v", err)
	}

	response := stripReasoning(strings.TrimSpace(completion.Choices[0].Content))
	return sanitizeSuggestedCorrespondent(response, suggestedTitle, content, availableCorrespondents), nil
}

// getSuggestedTags generates suggested tags for a document using the LLM
func (app *App) getSuggestedTags(
	ctx context.Context,
	content string,
	suggestedTitle string,
	availableTags []string,
	originalTags []string,
	logger *logrus.Entry) ([]string, error) {
	likelyLanguage := getLikelyLanguage()

	templateMutex.RLock()
	defer templateMutex.RUnlock()

	// Remove all paperless-gpt related tags from available tags
	availableTags = removeTagFromList(availableTags, manualTag)
	availableTags = removeTagFromList(availableTags, autoTag)
	availableTags = removeTagFromList(availableTags, autoOcrTag)

	// Get available tokens for content
	templateData := map[string]interface{}{
		"Language":      likelyLanguage,
		"AvailableTags": availableTags,
		"OriginalTags":  originalTags,
		"Title":         suggestedTitle,
		"CreateNewTags": createNewTags,
	}

	availableTokens, err := getAvailableTokensForContent(tagTemplate, templateData)
	if err != nil {
		logger.Errorf("Error calculating available tokens: %v", err)
		return nil, fmt.Errorf("error calculating available tokens: %v", err)
	}

	// Truncate content if needed
	truncatedContent, err := truncateContentByTokens(content, availableTokens)
	if err != nil {
		logger.Errorf("Error truncating content: %v", err)
		return nil, fmt.Errorf("error truncating content: %v", err)
	}

	// Execute template with truncated content
	var promptBuffer bytes.Buffer
	templateData["Content"] = truncatedContent
	err = tagTemplate.Execute(&promptBuffer, templateData)
	if err != nil {
		logger.Errorf("Error executing tag template: %v", err)
		return nil, fmt.Errorf("error executing tag template: %v", err)
	}

	prompt := promptBuffer.String()
	logger.Debugf("Tag suggestion prompt: %s", prompt)

	completion, err := app.LLM.GenerateContent(ctx, []llms.MessageContent{
		{
			Parts: []llms.ContentPart{
				llms.TextContent{
					Text: prompt,
				},
			},
			Role: llms.ChatMessageTypeHuman,
		},
	}, llms.WithTemperature(0))
	if err != nil {
		logger.Errorf("Error getting response from LLM: %v", err)
		return nil, fmt.Errorf("error getting response from LLM: %v", err)
	}

	response := stripReasoning(completion.Choices[0].Content)

	suggestedTags := strings.Split(response, ",")
	for i, tag := range suggestedTags {
		suggestedTags[i] = strings.TrimSpace(tag)
	}
	suggestedTags = sanitizeSuggestedTags(suggestedTags, suggestedTitle, content)

	// append the original tags to the suggested tags
	suggestedTags = append(suggestedTags, originalTags...)
	// Remove duplicates
	slices.Sort(suggestedTags)
	suggestedTags = slices.Compact(suggestedTags)
	suggestedTags = sanitizeSuggestedTags(suggestedTags, suggestedTitle, content)
	slices.Sort(suggestedTags)
	suggestedTags = slices.Compact(suggestedTags)

	// Filter out tags that are not in the available tags list (unless CREATE_NEW_TAGS is enabled)
	if createNewTags {
		// When creating new tags is enabled, keep all non-empty suggested tags
		filteredTags := []string{}
		for _, tag := range suggestedTags {
			if tag != "" {
				// Use the available tag's casing if it exists
				matched := false
				for _, availableTag := range availableTags {
					if strings.EqualFold(tag, availableTag) {
						filteredTags = append(filteredTags, availableTag)
						matched = true
						break
					}
				}
				if !matched {
					filteredTags = append(filteredTags, tag)
				}
			}
		}
		return filteredTags, nil
	}

	filteredTags := []string{}
	for _, tag := range suggestedTags {
		for _, availableTag := range availableTags {
			if strings.EqualFold(tag, availableTag) {
				filteredTags = append(filteredTags, availableTag)
				break
			}
		}
	}

	return filteredTags, nil
}

func sanitizeSuggestedTags(tags []string, title, content string) []string {
	forbiddenTags := map[string]struct{}{
		"rezerwacja": {},
		"rodo":       {},
		"szkoda":     {},
	}

	text := title + "\n" + content
	normalizedText := strings.ToLower(text)
	isPrivacyNotice := isPrivacyNoticeText(normalizedText)
	isInvoice := isInvoiceText(normalizedText)
	isTaxDocument := isTaxDocumentText(normalizedText)
	isOwnBusinessDocument := containsOwnBusiness(text)
	isInsuranceDocument := isInsuranceText(normalizedText)
	isVehicleDocument := isVehicleText(normalizedText)
	isReservationOrVoucher := isReservationOrVoucherText(normalizedText)
	isWithdrawalOrStatement := isWithdrawalOrStatementText(normalizedText)
	isBusinessITAgreement := isBusinessITAgreementText(normalizedText)
	isSignificantPurchase := isSignificantPurchaseText(normalizedText)
	isTelecomAgreement := isTelecomAgreementText(normalizedText)

	filteredTags := make([]string, 0, len(tags))
	for _, tag := range tags {
		normalizedTag := strings.ToLower(strings.TrimSpace(tag))
		if _, forbidden := forbiddenTags[normalizedTag]; forbidden {
			continue
		}
		if isPrivacyNotice {
			switch normalizedTag {
			case "dokumenty osobiste", "finanse", "umowy":
				continue
			}
		}
		if normalizedTag == "umowy" && (isReservationOrVoucher || isWithdrawalOrStatement) {
			continue
		}
		if normalizedTag == "podatki" && isInvoice && !isTaxDocument {
			continue
		}
		if normalizedTag == "zakupy" && isInvoice && !isSignificantPurchase {
			continue
		}
		if normalizedTag == "ubezpieczenia" && isTelecomAgreement {
			continue
		}
		if normalizedTag == "ubezpieczenia" && !isInsuranceDocument && !isReservationOrVoucher {
			continue
		}
		if normalizedTag == "zatrudnienie" && isBusinessITAgreement {
			continue
		}
		filteredTags = append(filteredTags, tag)
	}
	if isOwnBusinessDocument || isBusinessITAgreement {
		filteredTags = appendTagIfMissing(filteredTags, "Działalność gospodarcza")
	}
	if isInsuranceDocument {
		filteredTags = appendTagIfMissing(filteredTags, "Ubezpieczenia")
	}
	if isInsuranceDocument && isVehicleDocument {
		filteredTags = appendTagIfMissing(filteredTags, "Samochód")
	}
	if isSignificantPurchase {
		filteredTags = appendTagIfMissing(filteredTags, "Zakupy")
	}
	if isBusinessITAgreement {
		filteredTags = appendTagIfMissing(filteredTags, "Informatyka")
		filteredTags = appendTagIfMissing(filteredTags, "Umowy")
	}
	if isTelecomAgreement {
		filteredTags = appendTagIfMissing(filteredTags, "Telekomunikacja")
		filteredTags = appendTagIfMissing(filteredTags, "Umowy")
	}

	return filteredTags
}

func sanitizeSuggestedCorrespondent(suggested, title, content string, availableCorrespondents []string) string {
	correspondent := strings.TrimSpace(strings.Trim(suggested, "\""))
	if canonical := canonicalizeCorrespondent(correspondent, availableCorrespondents); canonical != "" {
		correspondent = canonical
	}

	normalizedText := strings.ToLower(title + "\n" + content)
	if !shouldRepairIssuerCorrespondent(correspondent, normalizedText) {
		return correspondent
	}

	if correspondent == "" || isUnknownCorrespondent(correspondent) || isOwnBusinessCorrespondent(correspondent) {
		if seller := extractDocumentIssuer(content, availableCorrespondents); seller != "" {
			return seller
		}
		if unknown := canonicalizeCorrespondent("Unknown", availableCorrespondents); unknown != "" {
			return unknown
		}
		return "Unknown"
	}

	return correspondent
}

func extractInvoiceSeller(content string, availableCorrespondents []string) string {
	return extractDocumentIssuer(content, availableCorrespondents)
}

func extractDocumentIssuer(content string, availableCorrespondents []string) string {
	for _, correspondent := range availableCorrespondents {
		if isUnknownCorrespondent(correspondent) || isOwnBusinessCorrespondent(correspondent) {
			continue
		}
		if strings.Contains(comparableName(content), comparableName(correspondent)) {
			return correspondent
		}
	}

	lines := strings.Split(content, "\n")
	for index, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		normalizedLine := strings.ToLower(trimmedLine)
		if !isSellerLabel(normalizedLine) {
			continue
		}

		if candidate := cleanCorrespondentCandidate(afterLabelValue(trimmedLine)); isUsableSellerCandidate(candidate) {
			if canonical := canonicalizeCorrespondent(candidate, availableCorrespondents); canonical != "" {
				return canonical
			}
			return candidate
		}

		for nextIndex := index + 1; nextIndex < len(lines) && nextIndex <= index+5; nextIndex++ {
			candidate := cleanCorrespondentCandidate(lines[nextIndex])
			normalizedCandidate := strings.ToLower(candidate)
			if isBuyerLabel(normalizedCandidate) {
				break
			}
			if isUsableSellerCandidate(candidate) {
				if canonical := canonicalizeCorrespondent(candidate, availableCorrespondents); canonical != "" {
					return canonical
				}
				return candidate
			}
		}
	}

	return ""
}

func appendTagIfMissing(tags []string, tag string) []string {
	if containsTag(tags, tag) {
		return tags
	}
	return append(tags, tag)
}

func containsTag(tags []string, expected string) bool {
	for _, tag := range tags {
		if strings.EqualFold(strings.TrimSpace(tag), expected) {
			return true
		}
	}
	return false
}

func containsOwnBusiness(text string) bool {
	return strings.Contains(strings.ToLower(text), ownBusinessName) ||
		strings.Contains(digitsOnly(text), ownBusinessNIP)
}

func isOwnBusinessCorrespondent(correspondent string) bool {
	normalized := strings.ToLower(correspondent)
	return strings.Contains(normalized, "aleksander goli") ||
		strings.Contains(digitsOnly(correspondent), ownBusinessNIP)
}

func isInvoiceText(normalizedText string) bool {
	return strings.Contains(normalizedText, "faktura") ||
		strings.Contains(normalizedText, "invoice")
}

func isPrivacyNoticeText(normalizedText string) bool {
	return strings.Contains(normalizedText, "przetwarzaniu danych osobowych") ||
		strings.Contains(normalizedText, "ochronie danych osobowych") ||
		strings.Contains(normalizedText, "polityka prywatności") ||
		strings.Contains(normalizedText, "polityka prywatnosci") ||
		strings.Contains(normalizedText, "privacy notice") ||
		strings.Contains(normalizedText, "privacy policy") ||
		containsSeparatedTerm(normalizedText, "rodo")
}

func isInsuranceText(normalizedText string) bool {
	return strings.Contains(normalizedText, "polisa") ||
		strings.Contains(normalizedText, "odszkodow") ||
		strings.Contains(normalizedText, "zgłoszenia szkody") ||
		strings.Contains(normalizedText, "zgloszenia szkody") ||
		strings.Contains(normalizedText, "formularz zgłoszenia szkody") ||
		strings.Contains(normalizedText, "formularz zgloszenia szkody") ||
		(strings.Contains(normalizedText, "pzu") && strings.Contains(normalizedText, "ubezpiecze"))
}

func isVehicleText(normalizedText string) bool {
	return strings.Contains(normalizedText, "pojazd") ||
		strings.Contains(normalizedText, "samochód") ||
		strings.Contains(normalizedText, "samochod") ||
		strings.Contains(normalizedText, "numer rejestracyjny") ||
		strings.Contains(normalizedText, "numer nadwozia") ||
		strings.Contains(normalizedText, "vin") ||
		strings.Contains(normalizedText, "marka i model")
}

func isReservationOrVoucherText(normalizedText string) bool {
	return strings.Contains(normalizedText, "voucher") ||
		isTravelReservationDocumentText(normalizedText)
}

func isTravelReservationDocumentText(normalizedText string) bool {
	if strings.Contains(normalizedText, "potwierdzenie rezerwacji") ||
		strings.Contains(normalizedText, "booking confirmation") ||
		strings.Contains(normalizedText, "reservation confirmation") {
		return true
	}
	if !strings.Contains(normalizedText, "rezerwac") &&
		!strings.Contains(normalizedText, "booking") &&
		!strings.Contains(normalizedText, "reservation") {
		return false
	}
	travelKeywords := []string{
		"podróż",
		"podroz",
		"hotel",
		"nocleg",
		"lot",
		"bilet",
		"wakac",
		"sun & fun",
		"travel",
		"hurghada",
	}
	for _, keyword := range travelKeywords {
		if strings.Contains(normalizedText, keyword) {
			return true
		}
	}
	return false
}

func isWithdrawalOrStatementText(normalizedText string) bool {
	return strings.Contains(normalizedText, "wypowiedzenie") ||
		strings.Contains(normalizedText, "odstąpienie") ||
		strings.Contains(normalizedText, "odstapienie") ||
		strings.Contains(normalizedText, "oświadczenie") ||
		strings.Contains(normalizedText, "oswiadczenie") ||
		strings.Contains(normalizedText, "declaration") ||
		strings.Contains(normalizedText, "statement")
}

func isBusinessITAgreementText(normalizedText string) bool {
	return strings.Contains(normalizedText, "atos") &&
		(strings.Contains(normalizedText, "poufno") ||
			strings.Contains(normalizedText, "non-disclosure") ||
			strings.Contains(normalizedText, "nda"))
}

func isTelecomAgreementText(normalizedText string) bool {
	return strings.Contains(normalizedText, "orange") &&
		strings.Contains(normalizedText, "umowa") &&
		(strings.Contains(normalizedText, "świadczenie usług") ||
			strings.Contains(normalizedText, "swiadczenie uslug") ||
			strings.Contains(normalizedText, "internet") ||
			strings.Contains(normalizedText, "telekomunikac"))
}

func isSignificantPurchaseText(normalizedText string) bool {
	if !isInvoiceText(normalizedText) && !strings.Contains(normalizedText, "paragon") {
		return false
	}
	purchaseKeywords := []string{
		"lantre",
		"macbook",
		"laptop",
		"komputer",
		"monitor",
		"iphone",
		"ipad",
		"sprzęt",
		"sprzet",
		"urządzenie",
		"urzadzenie",
	}
	for _, keyword := range purchaseKeywords {
		if strings.Contains(normalizedText, keyword) {
			return true
		}
	}
	return false
}

func shouldRepairIssuerCorrespondent(correspondent, normalizedText string) bool {
	if correspondent != "" && !isUnknownCorrespondent(correspondent) && !isOwnBusinessCorrespondent(correspondent) {
		return false
	}
	return isInvoiceText(normalizedText) ||
		isInsuranceText(normalizedText) ||
		isWithdrawalOrStatementText(normalizedText) ||
		isBusinessITAgreementText(normalizedText)
}

func isTaxDocumentText(normalizedText string) bool {
	taxKeywords := []string{
		"deklarac",
		"jpk",
		"pit",
		"cit",
		"urząd skarbow",
		"urzad skarbow",
		"rozliczenie podat",
		"decyzja podat",
		"podatek dochodowy",
		"vat-7",
		"vat 7",
	}
	for _, keyword := range taxKeywords {
		if strings.Contains(normalizedText, keyword) {
			return true
		}
	}
	return false
}

func isUnknownCorrespondent(correspondent string) bool {
	return strings.EqualFold(strings.TrimSpace(correspondent), "Unknown")
}

func canonicalizeCorrespondent(candidate string, availableCorrespondents []string) string {
	if strings.TrimSpace(candidate) == "" {
		return ""
	}
	normalizedCandidate := comparableName(candidate)
	for _, correspondent := range availableCorrespondents {
		normalizedCorrespondent := comparableName(correspondent)
		if normalizedCandidate == normalizedCorrespondent ||
			strings.Contains(normalizedCandidate, normalizedCorrespondent) ||
			strings.Contains(normalizedCorrespondent, normalizedCandidate) {
			return correspondent
		}
	}
	return ""
}

func isSellerLabel(normalizedLine string) bool {
	return strings.Contains(normalizedLine, "sprzedawca") ||
		strings.Contains(normalizedLine, "wystawca") ||
		strings.Contains(normalizedLine, "seller") ||
		strings.Contains(normalizedLine, "ubezpieczyciel") ||
		strings.Contains(normalizedLine, "towarzystwo") ||
		strings.Contains(normalizedLine, "usługodawca") ||
		strings.Contains(normalizedLine, "uslugodawca") ||
		strings.Contains(normalizedLine, "operator") ||
		strings.Contains(normalizedLine, "organizator")
}

func isBuyerLabel(normalizedLine string) bool {
	return strings.Contains(normalizedLine, "nabywca") ||
		strings.Contains(normalizedLine, "odbiorca") ||
		strings.Contains(normalizedLine, "buyer") ||
		strings.Contains(normalizedLine, "customer") ||
		strings.Contains(normalizedLine, "klient")
}

func afterLabelValue(line string) string {
	for _, separator := range []string{":", "-"} {
		if before, after, found := strings.Cut(line, separator); found && isSellerLabel(strings.ToLower(before)) {
			return after
		}
	}
	return ""
}

func cleanCorrespondentCandidate(candidate string) string {
	candidate = strings.TrimSpace(candidate)
	for _, marker := range []string{" NIP", " nip", " REGON", " regon", " VAT", " vat"} {
		if index := strings.Index(candidate, marker); index >= 0 {
			candidate = candidate[:index]
		}
	}
	return strings.Trim(candidate, " \t\r\n,.;:-")
}

func isUsableSellerCandidate(candidate string) bool {
	normalized := strings.ToLower(strings.TrimSpace(candidate))
	if len([]rune(normalized)) < 3 {
		return false
	}
	if isSellerLabel(normalized) || isBuyerLabel(normalized) || isOwnBusinessCorrespondent(normalized) {
		return false
	}
	excludedFragments := []string{"nip", "regon", "adres", "ul.", "kod pocztowy", "konto", "bank"}
	for _, fragment := range excludedFragments {
		if strings.Contains(normalized, fragment) {
			return false
		}
	}
	return true
}

func comparableName(text string) string {
	var builder strings.Builder
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			builder.WriteRune(r)
		}
	}
	return builder.String()
}

func containsSeparatedTerm(text, term string) bool {
	for _, field := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if field == term {
			return true
		}
	}
	return false
}

func digitsOnly(text string) string {
	var builder strings.Builder
	for _, r := range text {
		if unicode.IsDigit(r) {
			builder.WriteRune(r)
		}
	}
	return builder.String()
}

// getSuggestedDocumentType generates a suggested document type for a document using the LLM
func (app *App) getSuggestedDocumentType(
	ctx context.Context,
	content string,
	suggestedTitle string,
	availableDocumentTypes []string,
	logger *logrus.Entry) (string, error) {
	likelyLanguage := getLikelyLanguage()

	templateMutex.RLock()
	defer templateMutex.RUnlock()

	// Get available tokens for content
	templateData := map[string]interface{}{
		"Language":               likelyLanguage,
		"AvailableDocumentTypes": availableDocumentTypes,
		"Title":                  suggestedTitle,
	}

	availableTokens, err := getAvailableTokensForContent(documentTypeTemplate, templateData)
	if err != nil {
		logger.Errorf("Error calculating available tokens: %v", err)
		return "", fmt.Errorf("error calculating available tokens: %v", err)
	}

	// Truncate content if needed
	truncatedContent, err := truncateContentByTokens(content, availableTokens)
	if err != nil {
		logger.Errorf("Error truncating content: %v", err)
		return "", fmt.Errorf("error truncating content: %v", err)
	}

	// Execute template with truncated content
	var promptBuffer bytes.Buffer
	templateData["Content"] = truncatedContent
	err = documentTypeTemplate.Execute(&promptBuffer, templateData)
	if err != nil {
		logger.Errorf("Error executing document type template: %v", err)
		return "", fmt.Errorf("error executing document type template: %v", err)
	}

	prompt := promptBuffer.String()
	logger.Debugf("Document type suggestion prompt: %s", prompt)

	completion, err := app.LLM.GenerateContent(ctx, []llms.MessageContent{
		{
			Parts: []llms.ContentPart{
				llms.TextContent{
					Text: prompt,
				},
			},
			Role: llms.ChatMessageTypeHuman,
		},
	}, llms.WithTemperature(0))
	if err != nil {
		logger.Errorf("Error getting response from LLM: %v", err)
		return "", fmt.Errorf("error getting response from LLM: %v", err)
	}

	response := strings.TrimSpace(stripReasoning(completion.Choices[0].Content))
	response = sanitizeSuggestedDocumentType(response, suggestedTitle, content)

	// Validate that the response is in the available document types list
	for _, docType := range availableDocumentTypes {
		if strings.EqualFold(response, docType) {
			return docType, nil // Return the exact name from available types
		}
	}

	// If not found in available types, return empty string
	if response != "" {
		logger.Warnf("LLM suggested document type '%s' not found in available types, ignoring", response)
	}
	return "", nil
}

func sanitizeSuggestedDocumentType(suggested, title, content string) string {
	documentType := strings.TrimSpace(strings.Trim(suggested, "\""))
	normalizedText := strings.ToLower(title + "\n" + content)

	switch {
	case isThesisText(normalizedText):
		return "Praca dyplomowa"
	case strings.Contains(normalizedText, "voucher"):
		return "Voucher"
	case isTravelReservationDocumentText(normalizedText):
		return "Rezerwacja"
	case isTelecomAgreementText(normalizedText):
		return "Umowa"
	case isInsuranceText(normalizedText):
		return "Polisa"
	case isBusinessITAgreementText(normalizedText):
		return "Umowa"
	case isWithdrawalOrStatementText(normalizedText):
		return "Oświadczenie"
	case isInvoiceText(normalizedText):
		return "Faktura"
	case strings.Contains(normalizedText, "paragon"):
		return "Paragon"
	case isDiagnosticResultText(normalizedText):
		return "Wynik badania"
	case isPrivacyNoticeText(normalizedText):
		return "Dokument informacyjny"
	default:
		return documentType
	}
}

func isThesisText(normalizedText string) bool {
	return strings.Contains(normalizedText, "praca dyplomowa") ||
		strings.Contains(normalizedText, "praca inżynierska") ||
		strings.Contains(normalizedText, "praca inzynierska") ||
		strings.Contains(normalizedText, "praca magisterska")
}

func isDiagnosticResultText(normalizedText string) bool {
	return strings.Contains(normalizedText, "wynik badania") ||
		strings.Contains(normalizedText, "wyniki badania") ||
		strings.Contains(normalizedText, "wyniki badań") ||
		strings.Contains(normalizedText, "wyniki badan") ||
		strings.Contains(normalizedText, "laboratoryjn") ||
		strings.Contains(normalizedText, "diagnostyczn") ||
		strings.Contains(normalizedText, "vetlab")
}

// getSuggestedTitle generates a suggested title for a document using the LLM
func (app *App) getSuggestedTitle(ctx context.Context, content string, originalTitle string, logger *logrus.Entry) (string, error) {
	likelyLanguage := getLikelyLanguage()

	templateMutex.RLock()
	defer templateMutex.RUnlock()

	// Get available tokens for content
	templateData := map[string]interface{}{
		"Language": likelyLanguage,
		"Content":  content,
		"Title":    originalTitle,
	}

	availableTokens, err := getAvailableTokensForContent(titleTemplate, templateData)
	if err != nil {
		logger.Errorf("Error calculating available tokens: %v", err)
		return "", fmt.Errorf("error calculating available tokens: %v", err)
	}

	// Truncate content if needed
	truncatedContent, err := truncateContentByTokens(content, availableTokens)
	if err != nil {
		logger.Errorf("Error truncating content: %v", err)
		return "", fmt.Errorf("error truncating content: %v", err)
	}

	// Execute template with truncated content
	var promptBuffer bytes.Buffer
	templateData["Content"] = truncatedContent
	err = titleTemplate.Execute(&promptBuffer, templateData)

	if err != nil {
		return "", fmt.Errorf("error executing title template: %v", err)
	}

	prompt := promptBuffer.String()
	logger.Debugf("Title suggestion prompt: %s", prompt)

	completion, err := app.LLM.GenerateContent(ctx, []llms.MessageContent{
		{
			Parts: []llms.ContentPart{
				llms.TextContent{
					Text: prompt,
				},
			},
			Role: llms.ChatMessageTypeHuman,
		},
	}, llms.WithTemperature(0))
	if err != nil {
		return "", fmt.Errorf("error getting response from LLM: %v", err)
	}
	result := stripReasoning(completion.Choices[0].Content)
	return strings.TrimSpace(strings.Trim(result, "\"")), nil
}

// getSuggestedCreatedDate generates a suggested createdDate for a document using the LLM
func (app *App) getSuggestedCreatedDate(ctx context.Context, content string, logger *logrus.Entry) (string, error) {
	likelyLanguage := getLikelyLanguage()

	templateMutex.RLock()
	defer templateMutex.RUnlock()

	// Get available tokens for content
	templateData := map[string]interface{}{
		"Language": likelyLanguage,
		"Content":  content,
		"Today":    getTodayDate(), // must be in YYYY-MM-DD format
	}

	availableTokens, err := getAvailableTokensForContent(createdDateTemplate, templateData)
	if err != nil {
		logger.Errorf("Error calculating available tokens: %v", err)
		return "", fmt.Errorf("error calculating available tokens: %v", err)
	}

	// Truncate content if needed
	truncatedContent, err := truncateContentByTokens(content, availableTokens)
	if err != nil {
		logger.Errorf("Error truncating content: %v", err)
		return "", fmt.Errorf("error truncating content: %v", err)
	}

	// Execute template with truncated content
	var promptBuffer bytes.Buffer
	templateData["Content"] = truncatedContent
	err = createdDateTemplate.Execute(&promptBuffer, templateData)

	if err != nil {
		return "", fmt.Errorf("error executing createdDate template: %v", err)
	}

	prompt := promptBuffer.String()
	logger.Debugf("CreatedDate suggestion prompt: %s", prompt)

	completion, err := app.LLM.GenerateContent(ctx, []llms.MessageContent{
		{
			Parts: []llms.ContentPart{
				llms.TextContent{
					Text: prompt,
				},
			},
			Role: llms.ChatMessageTypeHuman,
		},
	}, llms.WithTemperature(0))
	if err != nil {
		return "", fmt.Errorf("error getting response from LLM: %v", err)
	}
	result := stripReasoning(completion.Choices[0].Content)
	return strings.TrimSpace(strings.Trim(result, "\"")), nil
}

// getSuggestedCustomFields generates suggested custom fields for a document using the LLM
func (app *App) getSuggestedCustomFields(ctx context.Context, doc Document, selectedFieldIDs []int, logger *logrus.Entry) ([]CustomFieldSuggestion, error) {
	// Fetch all available custom fields
	allCustomFields, err := app.Client.GetCustomFields(ctx)
	if err != nil {
		return nil, fmt.Errorf("error fetching all custom fields: %v", err)
	}

	// Filter to get only the selected custom fields
	var selectedCustomFields []CustomField
	for _, field := range allCustomFields {
		for _, selectedID := range selectedFieldIDs {
			if field.ID == selectedID {
				selectedCustomFields = append(selectedCustomFields, field)
				break
			}
		}
	}

	if len(selectedCustomFields) == 0 {
		return nil, nil // No fields to process
	}

	// Generate XML for the prompt
	var xmlBuilder strings.Builder
	xmlBuilder.WriteString("<custom_fields>\n")
	for _, field := range selectedCustomFields {
		xmlBuilder.WriteString(fmt.Sprintf("  <field name=\"%s\" type=\"%s\"></field>\n", field.Name, field.DataType))
	}
	xmlBuilder.WriteString("</custom_fields>")
	customFieldsXML := xmlBuilder.String()

	templateMutex.RLock()
	defer templateMutex.RUnlock()

	templateData := map[string]interface{}{
		"Language":        getLikelyLanguage(),
		"Title":           doc.Title,
		"CreatedDate":     doc.CreatedDate,
		"DocumentType":    doc.DocumentTypeName,
		"CustomFieldsXML": customFieldsXML,
	}

	availableTokens, err := getAvailableTokensForContent(customFieldTemplate, templateData)
	if err != nil {
		return nil, fmt.Errorf("error calculating available tokens for custom fields: %v", err)
	}

	truncatedContent, err := truncateContentByTokens(doc.Content, availableTokens)
	if err != nil {
		return nil, fmt.Errorf("error truncating content for custom fields: %v", err)
	}

	var promptBuffer bytes.Buffer
	templateData["Content"] = truncatedContent
	err = customFieldTemplate.Execute(&promptBuffer, templateData)
	if err != nil {
		return nil, fmt.Errorf("error executing custom field template: %v", err)
	}

	prompt := promptBuffer.String()
	logger.Debugf("Custom field suggestion prompt: %s", prompt)

	completion, err := app.LLM.GenerateContent(ctx, []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextContent{Text: prompt},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("error getting response from LLM for custom fields: %v", err)
	}

	response := stripReasoning(completion.Choices[0].Content)
	response = stripMarkdown(response)
	logger.Debugf("LLM response for custom fields: %s", response)

	// Temporary struct to unmarshal LLM response with field name
	type LLMCustomFieldResponse struct {
		Field string      `json:"field"`
		Value interface{} `json:"value"`
	}

	var llmSuggestedFields []LLMCustomFieldResponse
	// Handle empty or non-JSON response gracefully
	if strings.TrimSpace(response) == "" || !strings.HasPrefix(strings.TrimSpace(response), "[") {
		return []CustomFieldSuggestion{}, nil
	}

	err = json.Unmarshal([]byte(response), &llmSuggestedFields)
	if err != nil {
		logger.Errorf("Error unmarshalling custom fields JSON from LLM response: %v. Response: %s", err, response)
		return []CustomFieldSuggestion{}, nil // Return empty slice on parsing error
	}

	// Map field names back to IDs
	fieldNameIdMap := make(map[string]int)
	for _, field := range allCustomFields {
		fieldNameIdMap[field.Name] = field.ID
	}

	var finalSuggestedFields []CustomFieldSuggestion
	for _, llmField := range llmSuggestedFields {
		if id, ok := fieldNameIdMap[llmField.Field]; ok {
			finalSuggestedFields = append(finalSuggestedFields, CustomFieldSuggestion{
				ID:    id,
				Name:  llmField.Field,
				Value: llmField.Value,
			})
		} else {
			logger.Warnf("LLM returned unknown custom field name '%s', skipping.", llmField.Field)
		}
	}

	return finalSuggestedFields, nil
}

// generateDocumentSuggestions generates suggestions for a set of documents
func (app *App) generateDocumentSuggestions(ctx context.Context, suggestionRequest GenerateSuggestionsRequest, logger *logrus.Entry) ([]DocumentSuggestion, error) {
	// Fetch all available tags from paperless-ngx
	availableTagsMap, err := app.Client.GetAllTags(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch available tags: %v", err)
	}

	// Prepare a list of tag names
	availableTagNames := make([]string, 0, len(availableTagsMap))
	for tagName := range availableTagsMap {
		if tagName == manualTag {
			continue
		}
		availableTagNames = append(availableTagNames, tagName)
	}

	// Prepare a list of document correspodents
	availableCorrespondentsMap, err := app.Client.GetAllCorrespondents(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch available correspondents: %v", err)
	}

	// Prepare a list of correspondent names
	availableCorrespondentNames := make([]string, 0, len(availableCorrespondentsMap))
	for correspondentName := range availableCorrespondentsMap {
		availableCorrespondentNames = append(availableCorrespondentNames, correspondentName)
	}

	// Fetch all available document types from paperless-ngx
	availableDocumentTypes, err := app.Client.GetAllDocumentTypes(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch available document types: %v", err)
	}

	// Prepare a list of document type names
	availableDocumentTypeNames := make([]string, 0, len(availableDocumentTypes))
	for _, docType := range availableDocumentTypes {
		availableDocumentTypeNames = append(availableDocumentTypeNames, docType.Name)
	}

	documents := suggestionRequest.Documents
	documentSuggestions := []DocumentSuggestion{}

	var wg sync.WaitGroup
	var mu sync.Mutex
	errorsList := make([]error, 0)

	for i := range documents {
		wg.Add(1)
		go func(doc Document) {
			defer wg.Done()
			documentID := doc.ID
			docLogger := documentLogger(documentID)
			startTime := time.Now()
			docLogger.Printf("Processing Document ID %d...", documentID)

			content := doc.Content
			suggestedTitle := doc.Title
			var suggestedTags []string
			var suggestedCorrespondent string
			var suggestedDocumentType string
			var suggestedCreatedDate string
			var suggestedCustomFields []CustomFieldSuggestion

			if suggestionRequest.GenerateTitles {
				suggestedTitle, err = app.getSuggestedTitle(ctx, content, suggestedTitle, docLogger)
				if err != nil {
					mu.Lock()
					errorsList = append(errorsList, fmt.Errorf("Document %d: %v", documentID, err))
					mu.Unlock()
					docLogger.Errorf("Error processing document %d: %v", documentID, err)
					return
				}
			}

			if suggestionRequest.GenerateTags {
				suggestedTags, err = app.getSuggestedTags(ctx, content, suggestedTitle, availableTagNames, doc.Tags, docLogger)
				if err != nil {
					mu.Lock()
					errorsList = append(errorsList, fmt.Errorf("Document %d: %v", documentID, err))
					mu.Unlock()
					logger.Errorf("Error generating tags for document %d: %v", documentID, err)
					return
				}
			}

			if suggestionRequest.GenerateCorrespondents {
				suggestedCorrespondent, err = app.getSuggestedCorrespondent(ctx, content, suggestedTitle, availableCorrespondentNames, correspondentBlackList)
				if err != nil {
					mu.Lock()
					errorsList = append(errorsList, fmt.Errorf("Document %d: %v", documentID, err))
					mu.Unlock()
					log.Errorf("Error generating correspondents for document %d: %v", documentID, err)
					return
				}
			}

			if suggestionRequest.GenerateDocumentTypes {
				if len(availableDocumentTypeNames) == 0 {
					docLogger.Debug("Document type generation is enabled, but no document types are available in paperless-ngx.")
				} else {
					suggestedDocumentType, err = app.getSuggestedDocumentType(ctx, content, suggestedTitle, availableDocumentTypeNames, docLogger)
					if err != nil {
						mu.Lock()
						errorsList = append(errorsList, fmt.Errorf("Document %d: %v", documentID, err))
						mu.Unlock()
						log.Errorf("Error generating document type for document %d: %v", documentID, err)
						return
					}
				}
			}

			if suggestionRequest.GenerateCreatedDate {
				suggestedCreatedDate, err = app.getSuggestedCreatedDate(ctx, content, docLogger)
				if err != nil {
					mu.Lock()
					errorsList = append(errorsList, fmt.Errorf("Document %d: %v", documentID, err))
					mu.Unlock()
					log.Errorf("Error generating createdDate for document %d: %v", documentID, err)
					return
				}
			}

			if suggestionRequest.GenerateCustomFields {
				settingsMutex.RLock()
				selectedIDs := settings.CustomFieldsSelectedIDs
				settingsMutex.RUnlock()

				if len(selectedIDs) == 0 {
					log.Warnf("Custom field generation is enabled, but no custom fields are selected in the settings. Please select at least one custom field for this feature to work.")
				} else {
					suggestedCustomFields, err = app.getSuggestedCustomFields(ctx, doc, selectedIDs, docLogger)
					if err != nil {
						mu.Lock()
						errorsList = append(errorsList, fmt.Errorf("Document %d: %v", documentID, err))
						mu.Unlock()
						log.Errorf("Error generating custom fields for document %d: %v", documentID, err)
						return
					}
				}
			}

			mu.Lock()
			suggestion := DocumentSuggestion{
				ID:               documentID,
				OriginalDocument: doc,
			}
			settingsMutex.RLock()
			suggestion.CustomFieldsWriteMode = settings.CustomFieldsWriteMode
			suggestion.CustomFieldsEnable = settings.CustomFieldsEnable
			settingsMutex.RUnlock()
			// Titles
			if suggestionRequest.GenerateTitles {
				docLogger.Printf("Suggested title for document %d: %s", documentID, suggestedTitle)
				suggestion.SuggestedTitle = suggestedTitle
			} else {
				suggestion.SuggestedTitle = doc.Title
			}

			// Tags
			if suggestionRequest.GenerateTags {
				docLogger.Printf("Suggested tags for document %d: %v", documentID, suggestedTags)
				suggestion.SuggestedTags = suggestedTags
			} else {
				suggestion.SuggestedTags = doc.Tags
			}

			// Correspondents
			if suggestionRequest.GenerateCorrespondents {
				log.Printf("Suggested correspondent for document %d: %s", documentID, suggestedCorrespondent)
				suggestion.SuggestedCorrespondent = suggestedCorrespondent
			} else {
				suggestion.SuggestedCorrespondent = ""
			}

			// Document Type
			if suggestionRequest.GenerateDocumentTypes {
				log.Printf("Suggested document type for document %d: %s", documentID, suggestedDocumentType)
				suggestion.SuggestedDocumentType = suggestedDocumentType
			} else {
				suggestion.SuggestedDocumentType = ""
			}

			// CreatedDate
			if suggestionRequest.GenerateCreatedDate {
				log.Printf("Suggested createdDate for document %d: %s", documentID, suggestedCreatedDate)
				suggestion.SuggestedCreatedDate = suggestedCreatedDate
			} else {
				suggestion.SuggestedCreatedDate = ""
			}

			// Custom Fields
			if suggestionRequest.GenerateCustomFields {
				log.Printf("Suggested custom fields for document %d: %v", documentID, suggestedCustomFields)
				suggestion.SuggestedCustomFields = suggestedCustomFields
			}

			// Remove manual tag from the list of suggested tags
			suggestion.RemoveTags = []string{manualTag, autoTag}

			documentSuggestions = append(documentSuggestions, suggestion)
			mu.Unlock()

			elapsed := time.Since(startTime)
			// Format as HH:MM:SS using UTC zero-time base.
			runtime := time.Unix(0, elapsed.Nanoseconds()).UTC()
			docLogger.Printf("Document %d processed successfully. Runtime: %s",
				documentID, runtime.Format("15:04:05"))
		}(documents[i])
	}

	wg.Wait()

	if len(errorsList) > 0 {
		return nil, errorsList[0] // Return the first error encountered
	}

	return documentSuggestions, nil
}

// getTodayDate returns the current date in YYYY-MM-DD format
func getTodayDate() string {
	return time.Now().Format("2006-01-02")
}

// stripReasoning removes the reasoning from the content indicated by <think> and </think> tags.
func stripReasoning(content string) string {
	// Remove reasoning from the content
	reasoningStart := strings.Index(content, "<think>")
	if reasoningStart != -1 {
		reasoningEnd := strings.Index(content, "</think>")
		if reasoningEnd != -1 {
			content = content[:reasoningStart] + content[reasoningEnd+len("</think>"):]
		}
	}

	// Trim whitespace
	content = strings.TrimSpace(content)
	return content
}

// stripMarkdown removes the markdown code block from the content.
func stripMarkdown(content string) string {
	// Remove markdown code block
	if strings.HasPrefix(content, "```json") {
		content = strings.TrimPrefix(content, "```json")
		content = strings.TrimSuffix(content, "```")
	}
	return strings.TrimSpace(content)
}
