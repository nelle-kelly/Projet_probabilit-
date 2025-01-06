$(document).ready(function() {
    // Quand l'utilisateur choisit une loi et clique sur "Appliquer"
    $('#btn-appliquer').click(function() {
        var selectedLoi = $('#loi').val();

        // Masquer tous les groupes de paramètres
        $('.form-group').addClass('hidden');
        
        // Afficher les champs spécifiques à la loi sélectionnée
        if (selectedLoi === 'bernoulli') {
            $('.bernoulli').removeClass('hidden');
        } else if (selectedLoi === 'binomiale') {
            $('.binomiale').removeClass('hidden');
        } else if (selectedLoi === 'uniforme') {
            $('.uniforme').removeClass('hidden');
        } else if (selectedLoi === 'poisson') {
            $('.poisson').removeClass('hidden');
        } else if (selectedLoi === 'exponentielle') {
            $('.exponentielle').removeClass('hidden');
        } else if (selectedLoi === 'normale') {
            $('.normale').removeClass('hidden');
        }

        // Afficher le bouton "Calculer"
        $('#btn-calculer').show();
    });

    // Initialiser l'affichage
    $('#loi').trigger('change');
});
