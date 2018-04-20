$(document).ready(function(){
	//INDEX.HTML
	$('#loading').hide();
	$("#getstarted").click(function(event){
		$('#index-text').fadeOut( "slow", function() {
		});
		$('#index-img').fadeOut( "slow", function() {
			$('#index-form').fadeIn( "slow", function() {
			});
		});
	});

	$('#file').change(function() {
		$('#file-form').submit();
		$('#loading').show();
	});
	//RESULTS.HTML
	$("#analysisout").click(function(event){
		$('#miniinfo').fadeToggle();
	});
	

});
function reply_click(obj)
	{	console.log($(obj).parent())
    $(obj).parent()[0].classList.toggle("colG");
	}