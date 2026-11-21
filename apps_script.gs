// Google Sheet -> Extensions -> Apps Script -> paste this -> Deploy as Web app
const TOKEN = 'CHANGE_THIS_SECRET';   // same value goes in Streamlit secret SCRIPT_TOKEN
function doPost(e) {
  const d = JSON.parse(e.postData.contents);
  if (d.token !== TOKEN) return ContentService.createTextOutput('denied');
  const sh = SpreadsheetApp.getActive().getSheetByName('Learning');
  if (sh.getLastRow() === 0) sh.appendRow(['time','kind','sig','question','code','rating','note']);
  sh.appendRow([new Date(), d.kind, d.sig, d.question, d.code, d.rating, d.note]);
  return ContentService.createTextOutput('ok');
}
